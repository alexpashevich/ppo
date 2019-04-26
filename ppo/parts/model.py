import torch
import torch.nn as nn

from ppo.parts.distributions import Categorical, DiagGaussian
import ppo.tools.misc as misc

from bc.net.architectures import utils as bc_utils
from bc.dataset import Actions


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, **base_kwargs):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = ResnetBase(**base_kwargs)
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

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, memory_actions, rnn_hxs, masks, deterministic=False):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, actor_features, rnn_hxs = self.base(
            inputs,
            misc.dict_to_tensor(memory_actions)[0],
            misc.dict_to_tensor(rnn_hxs)[0],
            misc.dict_to_tensor(masks)[0])
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return (misc.tensor_to_dict(value, env_idxs)[0],
                misc.tensor_to_dict(action, env_idxs)[0],
                misc.tensor_to_dict(action_log_probs, env_idxs)[0],
                misc.tensor_to_dict(rnn_hxs, env_idxs)[0])

    def get_value_detached(self, inputs, actions, rnn_hxs, masks):
        inputs, env_idxs = misc.dict_to_tensor(inputs)
        value, _, _ = self.base(
            inputs,
            misc.dict_to_tensor(actions)[0],
            misc.dict_to_tensor(rnn_hxs)[0],
            misc.dict_to_tensor(masks)[0])
        return misc.tensor_to_dict(value.detach(), env_idxs)[0]

    def evaluate_actions(self, inputs, actions, rnn_hxs, masks, action):
        ''' This function is called from the PPO update so all the arguments are tensors. '''
        value, actor_features, rnn_hxs = self.base(inputs, actions, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class MasterPolicy(Policy):
    def __init__(self, obs_shape, action_space, **resnet_args):
        self.action_keys = Actions.action_space_to_keys(resnet_args['bc_args']['action_space'])[0]
        self.statistics = None
        super(MasterPolicy, self).__init__(obs_shape, action_space, **resnet_args)

    def get_worker_action(self, master_action, obs_dict):
        obs_tensor, env_idxs = misc.dict_to_tensor(obs_dict)
        master_action_filtered = []
        for env_idx in env_idxs:
            master_action_filtered.append(master_action[env_idx])
        master_action_filtered = torch.stack(master_action_filtered)
        action_tensor = self.base(obs_tensor, None, None, None, master_action=master_action_filtered)
        action_tensors_dict, env_idxs = misc.tensor_to_dict(action_tensor, env_idxs)
        action_dicts_dict = {}
        for env_idx, action_tensor in action_tensors_dict.items():
            action_dict = Actions.tensor_to_dict(action_tensor, self.action_keys, self.statistics)
            action_dict['skill'] = master_action[env_idx]
            action_dicts_dict[env_idx] = action_dict
        return action_dicts_dict, env_idxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self,
                 num_inputs,
                 recurrent_policy=False,
                 hidden_size=64,
                 action_memory=0,
                 **kwargs):
        super(MLPBase, self).__init__(recurrent_policy, num_inputs, hidden_size)

        if recurrent_policy:
            num_inputs = hidden_size

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

    def forward(self, inputs, actions, rnn_hxs, masks):
        if actions is not None:
            # agent has a memory
            x = torch.cat((inputs, actions), dim=1)
        else:
            # the input is the frames stack
            x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ResnetBase(NNBase):
    def __init__(
            self,
            num_skills=4,
            recurrent_policy=False,  # is not supported
            hidden_size=64,
            action_memory=0,
            bc_args=None,
            **unused_kwargs):
        super(ResnetBase, self).__init__(recurrent_policy, hidden_size, hidden_size)

        self.dim_action = bc_args['dim_action'] + 1
        self.num_skills = num_skills
        self.action_memory = action_memory
        self.resnet = bc_utils.make_resnet(
            archi=bc_args['archi'],
            mode='features',
            input_dim=bc_args['input_dim'])
        init_ = lambda m: misc.init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(bc_args['features_dim'] + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(bc_args['features_dim'] + action_memory, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        init_ = lambda m: misc.init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        skills_fc_sizes = [bc_args['features_dim'],
                           hidden_size,
                           hidden_size,
                           self.dim_action*bc_args['steps_action']]
        self.skills = []
        for skill in range(num_skills):
            skill_layers = []
            for i in range(len(skills_fc_sizes) - 1):
                skill_layers.append(nn.Linear(skills_fc_sizes[i], skills_fc_sizes[i + 1]))
                if i < len(skills_fc_sizes) - 2:
                    skill_layers.append(nn.ReLU())
            skill_layers = nn.Sequential(*skill_layers)
            self.skills.append(skill_layers)
        self.skills = nn.ModuleList(self.skills)

        self.train()

    def forward(self, inputs, actions, unused_rnn_hxs, unused_masks, master_action=None):
        # we now reshape the observations inside each environment
        # we do not use rnn_hxs but keep it for compatibility
        if actions is not None:
            # agent has a memory
            assert master_action is None
            features = self.resnet(inputs)
            features = torch.cat((features, actions), dim=1)
        else:
            # the input is the frames stack
            features = self.resnet(inputs)
        if master_action is None:
            # this is the policy step itself
            hidden_critic = self.critic(features.detach())
            hidden_actor = self.actor(features.detach())
            return self.critic_linear(hidden_critic), hidden_actor, unused_rnn_hxs
        else:
            # we want the skill actions
            # master_action: num_processes x 1
            skill_actions = []
            assert len(master_action) == len(features)
            for skill_id, feature in zip(master_action, features):
                skill_action = self.skills[skill_id](feature)[:self.dim_action]
                skill_actions.append(skill_action)
            return torch.stack(skill_actions)
