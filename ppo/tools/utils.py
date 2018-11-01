import torch
import torch.nn as nn

import os
import glob
import copy
import socket
import numpy as np

from ppo.tools.envs import VecNormalize, make_vec_envs

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


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


def set_up_training(args):
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)

def try_to_load_model(logdir):
    loaded, loaded_tuple = False, None
    for model_name in ('model_current.pt', 'model.pt'):
        try:
            loaded_tuple = torch.load(os.path.join(logdir, model_name))
            print('loaded a policy from {}'.format(os.path.join(logdir, model_name)))
            loaded = True
            break
        except Exception as e:
            # print('did not load a policy from {}'.format(os.path.join(logdir, model_name)))
            pass
    if not loaded:
        files = glob.glob(os.path.join(logdir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
        try:
            os.makedirs(os.path.join(logdir, '_eval'))
        except OSError:
            files = glob.glob(os.path.join(logdir, '_eval', '*.monitor.csv'))
            for f in files:
                os.remove(f)
    return loaded, loaded_tuple

def load_from_checkpoint(policy, path, device):
    if device.type == 'cpu':
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.load(path)
    state_dict = state_dict['net_state_dict']
    # BC training produces names of weights with "module." in the beginning
    state_dict_renamed = {}
    for key, value in state_dict.items():
        state_dict_renamed[key.replace('module.', '')] = value
    policy.base.resnet.load_state_dict(state_dict_renamed)
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

def close_envs(envs, close_envs_manually):
    envs.close()
    if close_envs_manually:
        for env in envs.venv.venv.envs:
            env.env.close()

def evaluate(policy, args_train, device, envs, render, eval_envs):
    args = copy.deepcopy(args_train)
    args.render = render
    num_processes = 1 if render else args.num_processes
    if eval_envs is None:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + num_processes, num_processes,
            args.gamma, None, args.add_timestep, device, True, env_config=args)
    if hasattr(policy.base, 'resnet'):
        # set the batch norm to the eval mode
        policy.base.resnet.eval()

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    returns_eval, lengths_eval = [], []
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    returns_current = np.array([0] * num_processes, dtype=np.float)
    lengths_current = np.array([0] * num_processes, dtype=np.float)

    while len(returns_eval) < args.num_eval_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obs
        # obs, reward, done, infos = eval_envs.step(action)
        if not args.use_bcrl_setup:
            obs, reward, done, infos = eval_envs.step(action)
        else:
            obs, reward, done, infos = do_master_step(
                action, obs, args.timescale, policy, eval_envs, render)

        returns_current, lengths_current = returns_current + reward[:, 0].numpy(), lengths_current + 1
        # append returns of envs that are done (reset)
        returns_eval.extend(returns_current[np.where(done)])
        lengths_eval.extend(lengths_current[np.where(done)])
        # zero out returns of the envs that are done (reset)
        returns_current[np.where(done)], lengths_current[np.where(done)] = 0, 0
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

    close_envs(eval_envs, close_envs_manually=render)
    if hasattr(policy.base, 'resnet'):
        # set the batch norm to the train mode
        policy.base.resnet.train()
    return eval_envs, returns_eval, lengths_eval


def do_master_step(
        master_action, master_obs, master_timescale, policy, envs, print_master_action=False):
    if print_master_action:
        print('master action = {}'.format(master_action.cpu().numpy()[:, 0]))
    master_reward = 0
    worker_obs = master_obs
    master_done = np.array([False] * master_action.shape[0])
    for _ in range(master_timescale):
        with torch.no_grad():
            worker_action = policy.get_worker_action(master_action, worker_obs)
        worker_obs, reward, done, infos = envs.step(worker_action)
        master_reward += reward
        master_done = np.logical_or(master_done, done)
        if done.any() and not done.all():
            # print('WARNING: one or several envs are done but not all during the macro step!')
            pass
        if master_done.all():
            break
    return worker_obs, master_reward, master_done, infos


def get_device(device):
    assert device in ('cpu', 'cuda'), 'device should be in (cpu, cuda)'
    if socket.gethostname() == 'gemini':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if device == 'cuda' else "cpu")
    return device
