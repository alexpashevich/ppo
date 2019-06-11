import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ppo.parts.distributions import Categorical, DiagGaussian
import ppo.tools.misc as misc

from bc.dataset import Actions
from bc.net.architectures.resnet import utils as resnet_utils


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MasterPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, bc_model, bc_statistics, **base_kwargs):
        super(MasterPolicy, self).__init__()

        self.action_keys = Actions.action_space_to_keys(base_kwargs['bc_args']['action_space'])[0]
        self.statistics = bc_statistics

        if len(obs_shape) == 3:
            self.base = ResnetBase(bc_model, **base_kwargs)
            # set the eval mode so the behavior of the skills is the same as in BC training
            self.base.resnet.eval()
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        # skill merging stuff
        self.skill_timesteps = np.zeros(base_kwargs['num_processes'], dtype=int)
        self.skill_timescales = base_kwargs['timescale']
        self.skill_mapping = base_kwargs['skills_mapping']

    def reset(self):
        self.skill_timesteps[:] = 0

    def report_skills_switch(self, skills_switched_array):
        for env_idx, skill_switched in enumerate(skills_switched_array):
            if skill_switched:
                self.skill_timesteps[env_idx] = 0

    def map_master_action(self, master_action, env_idxs):
        # master_action contains actions for each env, env_idxs cotains indices of envs in the batch
        master_action_real = []
        for env_idx, skill in master_action.items():
            if env_idx not in env_idxs:
                continue
            # get the real skill corresponding to the one given by the master
            env_mapping_list = self.skill_mapping[skill.item()]

            # check that the mime and master policy timescales are well synchronized
            env_timescale_merged_max = sum([
                self.skill_timescales[str(env_skill_real)] for env_skill_real in env_mapping_list])
            assert self.skill_timesteps[env_idx] < env_timescale_merged_max
            # print('skill ts[{}] = {}'.format(env_idx, self.skill_timesteps[env_idx]))

            # get the real action
            env_skill_counter = 0
            while True:
                env_timescale_merged = sum([
                    self.skill_timescales[str(env_skill_real)]
                    for env_skill_real in env_mapping_list[:env_skill_counter + 1]])
                if self.skill_timesteps[env_idx] < env_timescale_merged:
                    break
                else:
                    env_skill_counter += 1
                assert env_skill_counter < len(env_mapping_list)

            # save the new action in the correct format
            env_skill_real = env_mapping_list[env_skill_counter]
            master_action_real.append(torch.tensor([env_skill_real]).type_as(master_action[env_idx]))

            # increment the env timestep counter
            self.skill_timesteps[env_idx] += 1

        master_action_real = torch.stack(master_action_real)
        return master_action_real

    def get_worker_action(self, master_action, obs_dict):
        # print('action_master = {}'.format(
        #     [action.item() for env_idx, action in sorted(master_action.items())]))
        obs_tensor, env_idxs = misc.dict_to_tensor(obs_dict)
        master_action_real = self.map_master_action(master_action, env_idxs)
        # print('action_master_real = {}'.format(
        #     [action.item() for action in master_action_real]))
        action_tensor = self.base(obs_tensor, None, None, None, master_action=master_action_real)
        action_tensors_dict, env_idxs = misc.tensor_to_dict(action_tensor, env_idxs)
        action_tensors_dict_numpys = {key: value.cpu().numpy() for key, value in action_tensors_dict.items()}
        action_dicts_dict = {}
        master_action_real_dict, _ = misc.tensor_to_dict(master_action_real, env_idxs)
        for env_idx, action_tensor in action_tensors_dict_numpys.items():
            action_dict = Actions.tensor_to_dict(action_tensor, self.action_keys, self.statistics)
            action_dict['skill'] = master_action[env_idx].cpu().numpy()
            action_dict['skill_real'] = master_action_real_dict[env_idx].cpu().numpy()
            action_dicts_dict[env_idx] = action_dict
        return action_dicts_dict, env_idxs

    def act(self, inputs, memory_actions, deterministic=False):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, actor_features = self.base(
            inputs, misc.dict_to_tensor(memory_actions)[0])
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return (misc.tensor_to_dict(value, env_idxs)[0],
                misc.tensor_to_dict(action, env_idxs)[0],
                misc.tensor_to_dict(action_log_probs, env_idxs)[0])

    def get_value_detached(self, inputs, actions):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, _, _ = self.base(
            inputs, misc.dict_to_tensor(actions)[0])
        return misc.tensor_to_dict(value.detach(), env_idxs)[0]

    def evaluate_actions(self, inputs, actions_memory, action_master):
        ''' This function is called from the PPO update so all the arguments are tensors. '''
        value, actor_features = self.base(inputs, actions_memory)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action_master)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self,
                 num_inputs,
                 hidden_size=64,
                 action_memory=0,
                 **kwargs):
        super(MLPBase, self).__init__(hidden_size)

        init_ = lambda m: misc.init(m,
            misc.init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, actions_memory):
        if actions_memory is not None:
            # agent has a memory
            x = torch.cat((inputs, actions_memory), dim=1)
        else:
            # the input is the frames stack
            x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor


class ResnetBase(NNBase):
    def __init__(
            self,
            bc_model=None,
            # num_skills=4,
            skills_mapping=None,
            action_memory=0,
            bc_args=None,
            master_type='conv',
            master_num_channels=64,
            master_conv_filters=1,
            **unused_kwargs):
        super(ResnetBase, self).__init__(master_num_channels)

        self.dim_action = bc_args['dim_action'] + 1
        self.dim_action_seq = self.dim_action * bc_args['steps_action']
        self.num_skills = len(skills_mapping)
        self.action_memory = action_memory
        self.resnet = bc_model.net.module
        self.resnet.return_features = True

        # init_ = lambda m: misc.init(
        #     m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        if '/' in master_type:
            self.master_type, self.master_type_memory = master_type.split('/')
        else:
            self.master_type, self.master_type_memory = master_type, 'normal'

        assert self.master_type_memory in ('normal', 'extra_fc_concat', 'extra_fc_memory', 'conv_stack')
        if self.master_type_memory != 'normal':
            assert self.action_memory > 0

        if self.master_type_memory == 'extra_fc_concat':
            # apply 1 FC layer on top of the concatenation of skills and memory
            self.actor_extra_fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._hidden_size + self.action_memory, master_num_channels))
            self.critic_extra_fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._hidden_size + self.action_memory, master_num_channels))
        if self.master_type_memory == 'extra_fc_memory':
            # apply 1 FC layer on top of the memory and concatenate it with the vision output
            self.memory_extra_fc = nn.Sequential(
                nn.Linear(self.action_memory, self.action_memory),
                nn.ReLU())

        inplanes = bc_args['features_dim']
        if self.master_type_memory == 'conv_stack':
            # make a tensor of action_memory x 7 x 7 and stack it to the conv activations
            inplanes += self.action_memory

        self.actor, self.critic = [self._create_head(
            master_type=self.master_type,
            num_skills=self.num_skills,
            num_channels=master_num_channels,
            inplanes=inplanes,
            size_conv_filters=master_conv_filters) for _ in range(2)]

        init_ = lambda m: misc.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

        self.train()

    def _create_head(self, master_type, num_skills, num_channels, inplanes, size_conv_filters):
        head_conv, head_fc = resnet_utils.make_master_head(
            master_head_type=master_type,
            num_skills=num_skills,
            num_channels=num_channels,
            inplanes=inplanes,
            size_conv_filters=size_conv_filters)
        if master_type == 'fc':
            return head_fc
        else:
            return head_conv

    @property
    def output_size(self):
        if self.master_type_memory in ('normal', 'extra_fc_memory'):
            return self._hidden_size + self.action_memory
        else:
            return self._hidden_size

    def forward(self, inputs, actions_memory, master_action=None):
        # we now reshape the observations inside each environment
        skills_actions, master_features = self.resnet(inputs)

        if master_action is None:
            # this is the policy step itself
            if self.master_type_memory == 'conv_stack':
                actions_memory = actions_memory[..., None, None].repeat(1, 1, 7, 7)
                master_features = torch.cat((master_features, actions_memory), dim=1)
            hidden_critic = self.critic(master_features.detach())
            hidden_actor = self.actor(master_features.detach())
            if self.master_type != 'fc':
                for tensor in (hidden_actor, hidden_critic):
                    # we assume that both tensors are 7x7 conv activations
                    assert tensor.shape[2] == 7 and tensor.shape[3] == 7
                hidden_critic = F.avg_pool2d(hidden_critic, hidden_critic.shape[-1])[..., 0, 0]
                hidden_actor = F.avg_pool2d(hidden_actor, hidden_actor.shape[-1])[..., 0, 0]
            if actions_memory is not None and self.master_type_memory != 'conv_stack':
                actions_memory = 2 * actions_memory / (self.num_skills - 1) - 1
                if self.master_type_memory == 'extra_fc_memory':
                    actions_memory = self.memory_extra_fc(actions_memory)
                hidden_critic = torch.cat((hidden_critic, actions_memory), dim=1)
                hidden_actor = torch.cat((hidden_actor, actions_memory), dim=1)
            if self.master_type_memory == 'extra_fc_concat':
                hidden_critic = self.critic_extra_fc(hidden_critic)
                hidden_actor = self.actor_extra_fc(hidden_actor)

            return self.critic_linear(hidden_critic), hidden_actor
        else:
            # we want the skill actions
            # master_action: num_processes x 1
            skill_actions = []
            assert len(master_action) == len(master_features)
            for env_idx, skill_id in enumerate(master_action):
                skill_action = skills_actions[
                    env_idx,
                    self.dim_action_seq * skill_id: self.dim_action_seq * skill_id + self.dim_action]
                skill_actions.append(skill_action)
            return torch.stack(skill_actions)
