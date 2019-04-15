import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space,
                 recurrent_hidden_state_size,
                 action_memory=0):
        num_steps *= num_processes
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.action_memory = action_memory
        self.steps = np.zeros(num_processes, dtype=np.int32)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self,
               obs,
               recurrent_hidden_states,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks,
               indices=None):
        if indices is None:
            indices = np.range(self.num_processes)
        for index in indices:
            step_value = self.steps[index]
            self.obs[step_value + 1, index].copy_(obs[index])
            self.recurrent_hidden_states[step_value + 1, index].copy_(recurrent_hidden_states[index])
            self.actions[step_value, index].copy_(actions[index])
            self.action_log_probs[step_value, index].copy_(action_log_probs[index])
            self.value_preds[step_value, index].copy_(value_preds[index])
            self.rewards[step_value, index].copy_(rewards[index])
            self.masks[step_value + 1, index].copy_(masks[index])
            self.steps[index] = (self.steps[index] + 1) % self.num_steps

    def get_last(self, tensor, *args, **kwargs):
        if tensor is self.actions:
            return self._get_last_actions(*args, **kwargs)
        lasts = []
        for index in range(tensor.shape[1]):
            lasts.append(tensor[self.steps[index], index])
        return torch.stack(lasts)

    def _get_last_actions(self, steps=None, processes=None):
        if self.action_memory == 0:
            return None
        if processes is None:
            processes = np.arange(self.num_processes)
        if steps is None:
            steps = self.steps[processes]
        last_actions = -torch.ones(len(processes), self.action_memory).type_as(self.obs)
        for idx, (step, process) in enumerate(zip(steps, processes)):
            process_resets = np.where(self.masks[:step + 1, process, 0] == 0)[0]
            if process_resets.shape[0] > 0:
                last_reset = process_resets.max()
            else:
                last_reset = 0
            actions_available = np.clip(step - last_reset, 0, self.action_memory)
            # print('env step = {}, available = {}, last_reset = {}'.format(
            #     process, actions_available, last_reset))
            if actions_available > 0:
                last_actions_ = self.actions[step - actions_available: step, process, 0]
                last_actions[idx, -actions_available:] = last_actions_
        # print('last_actions = {}'.format(last_actions))
        return last_actions

    def after_update(self):
        last_indices = np.stack((self.steps, np.arange(self.steps.shape[0])))
        self.obs[0].copy_(self.obs[last_indices])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[last_indices])
        self.masks[0].copy_(self.masks[last_indices])
        self.steps = np.zeros_like(self.steps)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            raise NotImplementedError
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            for env_idx in range(next_value.shape[0]):
                self.returns[self.steps[env_idx], env_idx] = next_value[env_idx]
                for step in reversed(range(self.steps[env_idx])):
                    self.returns[step, env_idx] = self.returns[step + 1, env_idx] * \
                                         gamma * self.masks[step + 1, env_idx] + \
                                         self.rewards[step, env_idx]

    def feed_forward_generator(self, advantages, num_mini_batch):
        batch_size = int(np.sum(self.steps))
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of env steps ({}) "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(batch_size, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        # get the (i, j) indices of the filled transitions in the right order
        transitions_ordered_indices = np.concatenate(
            [np.stack((range(s), [i] * s)) for i, s in enumerate(self.steps) if s > 0], axis=1)
        for indices in sampler:
            # replace the batch indices i by rollouts indices (i, j)
            indices = transitions_ordered_indices[:, indices]
            obs_batch = self.obs[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1][indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            masks_batch = self.masks[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            adv_targ = advantages[indices]

            timesteps, env_idxs = indices
            last_actions_batch = self._get_last_actions(timesteps, env_idxs)
            yield obs_batch, last_actions_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        raise NotImplementedError
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
