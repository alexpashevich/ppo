import torch
import torch.nn as nn

import os
import glob
import copy
import socket
import numpy as np

from ppo.tools.envs import VecNormalize, VecPyTorchFrameStack, make_vec_envs
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import ppo.tools.stats as stats
import ppo.tools.gifs as gifs

from PIL import Image

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
    loaded, loaded_tuple = False, None
    for model_name in ('model_current.pt', 'model.pt'):
        try:
            loaded_tuple = torch.load(os.path.join(logdir, model_name))
            print('loaded a policy from {}'.format(os.path.join(logdir, model_name)))
            loaded = True
            break
        except Exception as e:
            pass
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

def evaluate(policy, args_train, device, train_envs_or_ob_rms, eval_envs, env_render=None):
    args = copy.deepcopy(args_train)
    args.render = False
    # make the evaluation horizon longer (if eval_max_length_factor > 1)
    args.max_length = int(args.max_length * args.eval_max_length_factor)
    num_processes = args.num_eval_episodes
    if eval_envs is None:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + num_processes, num_processes,
            args.gamma, args.add_timestep, device, True, env_config=args)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        if 'RunningMeanStd' in str(type(train_envs_or_ob_rms)):
            ob_rms = train_envs_or_ob_rms
        else:
            ob_rms = get_vec_normalize(train_envs_or_ob_rms).ob_rms
        vec_norm.ob_rms = ob_rms

    obs = eval_envs.reset()
    if env_render:
        env_render.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    stats_global, stats_local = stats.init(num_processes, eval=True)

    if args.save_gifs:
        gifs_global, gifs_local = gifs.init(num_processes)
    else:
        gifs_global = None

    print('Evaluating...')
    while len(stats_global['return']) < args.num_eval_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obs
        master_step_output = do_master_step(
            action, obs, args.timescale, policy, eval_envs,
            hrlbc_setup=args.hrlbc_setup,
            env_render=env_render,
            return_observations=args.save_gifs)
        obs, reward, done, infos = master_step_output[:4]
        if args.save_gifs:
            # saving gifs only works for the BCRL setup
            gifs_global, gifs_local = gifs.update(
                gifs_global, gifs_local, action, done, stats_local['done_before'],
                master_step_output[4])

        stats_global, stats_local = stats.update(
            stats_global, stats_local, reward, done, infos, args, overwrite_terminated=False)
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
    return eval_envs, stats_global, gifs_global

def do_master_step(
        master_action, master_obs, master_timescale, policy, envs,
        hrlbc_setup=False, env_render=None, return_observations=False):
    print('master action = {}'.format(master_action[:, 0]))
    master_reward = 0
    skill_obs = master_obs
    if return_observations:
        envs_history = {'observations': [[] for _ in range(master_action.shape[0])],
                        'skill_actions': [[] for _ in range(master_action.shape[0])]}
    master_done = np.array([False] * master_action.shape[0])
    for _ in range(master_timescale):
        if hrlbc_setup:
            # get the skill action
            with torch.no_grad():
                skill_action = policy.get_worker_action(master_action, skill_obs)
        else:
            # it is not really a skill action, but we use this name to simplify the code
            skill_action = master_action
        for env_id, done_before in enumerate(master_done):
            # might be not the beget the skill action
            # we apply the null action to it
            if done_before:
                skill_action[env_id] = 0
        skill_obs, reward, done, infos = envs.step(skill_action)
        if env_render is not None:
            env_render.step(skill_action[:1].cpu().numpy())
        if return_observations:
            for env_id, (done_before, done_now) in enumerate(zip(master_done, done)):
                # we do not want to record gifs after resets
                if not done_before and not done_now:
                    envs_history['observations'][env_id].append(skill_obs[env_id].cpu().numpy())
                    envs_history['skill_actions'][env_id].append(skill_action[env_id].cpu().numpy())
                    # skill_actions_envs_list.append(skill_action.cpu().numpy())
        # we do not add the rewards after reset
        reward[np.where(master_done)] = 0
        master_reward += reward
        master_done = np.logical_or(master_done, done)
        if master_done.all():
            break
    if not return_observations:
        return skill_obs, master_reward, master_done, infos
    else:
        return skill_obs, master_reward, master_done, infos, envs_history

def reset_early_terminated_envs(envs, env_render, done, obs, device, num_frames=3):
    # TODO: replace this method with a flexible timescale
    # we use the master with a fixed timescale, so some envs receive the old master action after reset
    # we manually reset the envs that were terminated during the master step after it
    done_idxs = np.where(done)[0]
    if isinstance(envs, SubprocVecEnv):
        # we have several envs
        remotes = envs.venv.venv.remotes
        for idx in done_idxs:
            remotes[idx].send(('reset', None))
    if env_render and done[0]:
        env_render.reset()
    if isinstance(envs, SubprocVecEnv):
        for idx in done_idxs:
            # we have several envs
            obs_numpy = remotes[idx].recv()
            obs_torch = torch.from_numpy(obs_numpy).float().to(device)
            if isinstance(envs, VecPyTorchFrameStack):
                for _ in range(num_frames):
                    envs.stacked_obs[idx].append(obs_torch)
            else:
                obs[idx] = obs_torch
        if isinstance(envs, VecPyTorchFrameStack):
            obs = envs._deque_to_tensor()
    else:
        # DummyVecEnv is used, we have only one env
        if 0 in done_idxs:
            obs = envs.reset()
    return obs