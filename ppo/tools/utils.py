import torch
import torch.nn as nn

import os
import glob
import copy
import socket
import numpy as np

from ppo.tools.envs import VecNormalize, make_vec_envs

from PIL import Image
from bc.utils import videos

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

def evaluate(policy, args_train, device, envs, eval_envs, env_render=None):
    args = copy.deepcopy(args_train)
    args.render = False
    if eval_envs is None:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + args.num_processes, args.num_processes,
            args.gamma, args.add_timestep, device, True, env_config=args)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    returns_eval, lengths_eval = [], []
    obs = eval_envs.reset()
    if env_render:
        env_render.reset()
    eval_recurrent_hidden_states = torch.zeros(
        args.num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)
    returns_current = np.array([0] * args.num_processes, dtype=np.float)
    lengths_current = np.array([0] * args.num_processes, dtype=np.float)

    frames = [[] for _ in range(args.num_processes)]
    skill_actions = [[] for _ in range(args.num_processes)]
    master_actions = [[] for _ in range(args.num_processes)]
    idx_video = 0
    gifs_dir = os.path.join(args.logdir, args.timestamp, 'gifs')
    if not os.path.exists(gifs_dir):
        os.mkdir(gifs_dir)

    print('Evaluating...')
    while len(returns_eval) < args.num_eval_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obs
        # obs, reward, done, infos = eval_envs.step(action)
        if not args.use_bcrl_setup:
            obs, reward, done, infos = eval_envs.step(action)
            if env_render is not None:
                env_render.step(action[:1].numpy())
        else:
            stack_obs, stack_act, reward, done, infos = do_master_step(
                action, obs, args.timescale, policy, eval_envs, env_render)
            obs = stack_obs[-1]

        for obs_skill, act_skill in zip(stack_obs, stack_act):
            for i, (obs_worker, act_worker) in enumerate(zip(obs_skill, act_skill)):
                frame = np.array((0.5+obs_worker.cpu().numpy()*0.5)*255, dtype=np.uint8)
                frames[i].append(frame)
                skill_actions[i].append(act_worker)
        for i, action_worker in enumerate(action.cpu().numpy()):
            master_actions[i].append(action_worker[0])

        idxs_done = np.where(done)[0]
        for idx in idxs_done:
            print('Writing video {}'.format(idx_video))
            print('Master actions {}'.format(master_actions[idx]))
            gif_name = '{:02}.gif'.format(idx_video)
            videos.write_video(frames[idx], os.path.join(gifs_dir, gif_name))
            np.savez(os.path.join(gifs_dir, '{:02}.npz'.format(idx_video)), actions=skill_actions[idx])
            idx_video += 1
            frames[idx] = []
            master_actions[idx] = []
            skill_actions[idx] = []

        returns_current, lengths_current = returns_current + reward[:, 0].numpy(), lengths_current + 1
        # append returns of envs that are done (reset)
        returns_eval.extend(returns_current[np.where(done)])
        lengths_eval.extend(lengths_current[np.where(done)])
        # zero out returns of the envs that are done (reset)
        returns_current[np.where(done)], lengths_current[np.where(done)] = 0, 0
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

    return eval_envs, returns_eval, lengths_eval


def do_master_step(
        master_action, master_obs, master_timescale, policy, envs, env_render=None):
    # TODO: do we want to print it? (all the master actions)
    print_master_action = False
    if print_master_action:
        if hasattr(master_action, 'cpu'):
            master_action = master_action.cpu().numpy()[:, 0]
        print('master action = {}'.format(master_action))
    master_reward = 0
    worker_obs = master_obs
    stack_obs = [master_obs]
    stack_act = []
    master_done = np.array([False] * master_action.shape[0])
    for _ in range(master_timescale):
        with torch.no_grad():
            worker_action = policy.get_worker_action(master_action, worker_obs)
        worker_obs, reward, done, infos = envs.step(worker_action)
        stack_obs.append(worker_obs)
        stack_act.append(worker_action.cpu().numpy())
        if env_render is not None:
            env_render.step(worker_action[:1].numpy())
        master_reward += reward
        master_done = np.logical_or(master_done, done)
        if done.any() and not done.all():
            # print('WARNING: one or several envs are done but not all during the macro step!')
            pass
        if master_done.all():
            break
    return stack_obs, stack_act, master_reward, master_done, infos


def get_device(device):
    assert device in ('cpu', 'cuda'), 'device should be in (cpu, cuda)'
    if socket.gethostname() == 'gemini' or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if device == 'cuda' else "cpu")
    return device
