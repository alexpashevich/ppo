import torch
import torch.nn as nn

import os
import socket

from ppo.tools.envs import VecNormalize


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)
    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def seed_torch(args):
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)

def get_device(device):
    assert device in ('cpu', 'cuda'), 'device should be in (cpu, cuda)'
    if socket.gethostname() == 'gemini' or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if device == 'cuda' else "cpu")
    return device

def try_to_load_model(logdir):
    loaded_tuple = None
    for model_name in ('model_current.pt', 'model.pt'):
        try:
            loaded_tuple = torch.load(os.path.join(logdir, model_name))
            print('loaded a policy from {}'.format(os.path.join(logdir, model_name)))
            break
        except Exception as e:
            pass
    return loaded_tuple

# dirty stuff, TODO: refactor later
def map_state_dict_key(key, num_skills):
    fcs_idx = int(key.replace('base.resnet.fcs.', '').split('.')[0])
    layer_idx, suffix = key.replace('base.resnet.fcs.', '').split('.')[1:]
    layer_idx = int(layer_idx)
    if fcs_idx < num_skills:
        # this is a part of the skills
        if layer_idx in {0, 2}:
            # first two fc layers, should go to the hidden actor
            key_new = 'base.actor_skills.{}.{}.{}'.format(fcs_idx, layer_idx + 1, suffix)
        else:
            # layer layer, should go to the dist
            key_new = 'dist_skills.{}.fc_mean.{}'.format(fcs_idx, suffix)
    else:
        # this is a part of the master
        if layer_idx in {0, 2}:
            # first two fc layers, should go to the hidden actor
            key_new = 'base.actor.{}.{}'.format(layer_idx + 1, suffix)
        else:
            # layer layer, should go to the dist
            key_new = 'dist.linear.{}'.format(suffix)
    return key_new


def load_from_checkpoint(policy, path, device, learn_skills=False):
    if device.type == 'cpu':
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.load(path)
    state_dict = state_dict['net_state_dict']
    # BC training produces names of weights with "module." in the beginning
    state_dict_renamed = {}
    for key, value in state_dict.items():
        state_dict_renamed[key.replace('module.', 'base.resnet.')] = value
    if learn_skills:
        state_dict_correct = {}
        for key, value in state_dict_renamed.items():
            if 'fcs' not in key:
                key_new = key
            else:
                key_new = map_state_dict_key(key, policy.base.num_skills)
            if 'dist_skills' in key_new:
                skill_action_dim = policy.dist_skills[0].fc_mean.weight.shape[0]
                value = value[:skill_action_dim]
            state_dict_correct[key_new] = value
        state_dict_renamed = state_dict_correct

    # policy.base.resnet.load_state_dict(state_dict_renamed, strict=not learn_skills)
    # TODO: the strict param is for critics and the dist_skills std, remove it later
    policy.load_state_dict(state_dict_renamed, strict=not learn_skills)
    print('loaded the BC checkpoint from {}'.format(path))

def load_optimizer(optimizer, optimizer_state_dict, device):
    optimizer.load_state_dict(optimizer_state_dict)
    target_device = 'cpu' if device.type == 'cpu' else 'cuda'
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = getattr(v, target_device)()

def load_ob_rms(ob_rms, envs):
    if ob_rms:
        try:
            get_vec_normalize(envs).ob_rms = ob_rms
        except:
            print('WARNING: did not manage to reuse the normalization statistics')
