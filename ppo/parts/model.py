import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ppo.parts.distributions import Categorical, DiagGaussian
from ppo.tools.utils import init, init_normc_

from bc.net.architectures import resnet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            # self.base = CNNBase(obs_shape[0], **base_kwargs)
            self.base = ResnetBase(obs_shape[0], **base_kwargs)
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

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class MasterPolicy(Policy):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(MasterPolicy, self).__init__(obs_shape, action_space, base_kwargs)

    def get_worker_action(self, master_action, observation):
        worker_action = self.base(observation, None, None, master_action=master_action)
        return worker_action


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


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, **kwargs):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent_policy=False, hidden_size=64, **kwargs):
        super(MLPBase, self).__init__(recurrent_policy, num_inputs, hidden_size)

        if recurrent_policy:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ResnetBase(NNBase):
    def __init__(
            self,
            num_inputs,
            num_skills=4,
            dim_skill_action=4,
            num_skill_action_pred=1,
            recurrent_policy=False,  # is not supported
            hidden_size=64,
            archi='resnet18',
            pretrained=False,
            **kwargs):
        super(ResnetBase, self).__init__(recurrent_policy, hidden_size, hidden_size)

        self.num_skills = num_skills
        self.dim_skill_action = dim_skill_action
        self.num_skill_action_pred = num_skill_action_pred
        num_outputs_resnet = self.num_skills * dim_skill_action * num_skill_action_pred
        if kwargs['use_direct_actions']:
            num_outputs_resnet = dim_skill_action
        self.resnet = getattr(resnet, archi)(
            pretrained=pretrained,
            input_dim=num_inputs,
            num_classes=num_outputs_resnet,  # dim_action in ResNet
            num_skills=num_skills,
            dim_action=dim_skill_action*num_skill_action_pred,  # dim_action in ResNetBranch
            return_features=True)

        self._transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize([224, 224]),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
                                              ])

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            Flatten(),
            init_(nn.Linear(512, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            Flatten(),
            init_(nn.Linear(512, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, unused_rnn_hxs, unused_masks, master_action=None):
        # inputs_reshaped = inputs_reshaped.type_as(inputs)
        # we now reshape the observations inside each environment, remove this code later
        # inputs_reshaped = []
        # for depth_maps_stack in inputs:
        #     depth_maps_reshaped_stack = []
        #     for depth_map in depth_maps_stack:
        #         # 1) transforms.ToPILImage expects 3D tensor, so we use [None]
        #         # 2) transforms.ToPILImage expects image between 0. and 1.
        #         # 3) we remove the extra dim with [0] afterwards
        #         depth_maps_reshaped_stack.append(self._transform(depth_map.cpu()[None] / 255)[0])
        #     inputs_reshaped.append(torch.stack(depth_maps_reshaped_stack))
        # inputs_reshaped = torch.stack(inputs_reshaped)
        # we do not use rnn_hxs but keep it for compatibility
        all_skills_actions, features = self.resnet(inputs)
        if master_action is None:
            # this is the policy step itself
            hidden_critic = self.critic(features.detach())
            hidden_actor = self.actor(features.detach())
            return self.critic_linear(hidden_critic), hidden_actor, unused_rnn_hxs
        else:
            # all_skills_actions: num_processes x (num_skills*dim_skill_action*num_skill_action_pred)
            # master_action: num_processes x 1
            skills_actions = []
            for env_id, skill_id in enumerate(master_action):
                # for the process I to access the skill J action we need the slice
                # proc_I_skill_J action beginning is at:
                # all_skills_actions[I, J x dim_skill_action x num_skill_action_pred]
                skill_action_begin = skill_id * self.dim_skill_action * self.num_skill_action_pred
                # proc_I_skill_J action end is at:
                # all_skills_actions[I, J x dim_skill_action x num_skill_action_pred + dim_skill_action]
                skill_action_end = skill_action_begin + self.dim_skill_action
                # so the proc_I_skill_J action is at
                # all_skills_actions[I, J x dim_skill_action x num_skill_action_pred:
                #                       J x dim_skill_action x num_skill_action_pred + dim_skill_action]
                skill_action = all_skills_actions[env_id, skill_action_begin:skill_action_end]
                skills_actions.append(skill_action)
            skills_actions = torch.stack(skills_actions)
            return skills_actions

