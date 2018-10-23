import torch
import torch.nn as nn

import os
import glob
import copy

from envs import VecNormalize, make_vec_envs

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
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    logdir = os.path.join(args.logdir, args.timestamp)
    eval_logdir = os.path.join(logdir, "_eval")
    return device, logdir, eval_logdir

def try_to_load_model(logdir, eval_logdir):
    policy, optimizer_dict, ob_rms, epoch = None, None, None, None
    loaded = False
    for model_name in ('model_current.pt', 'model.pt'):
        try:
            policy, optimizer_dict, ob_rms, epoch = torch.load(os.path.join(logdir, model_name))
            print('loaded a policy from {}'.format(os.path.join(logdir, model_name)))
            loaded = True
            break
        except Exception as e:
            print('did not load a policy from {}'.format(os.path.join(logdir, model_name)))
    if not loaded:
        print('not loaded a policy, cleaning the logdir {}'.format(logdir))
        files = glob.glob(os.path.join(logdir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
        try:
            os.makedirs(eval_logdir)
        except OSError:
            files = glob.glob(os.path.join(eval_logdir, '*.monitor.csv'))
            for f in files:
                os.remove(f)
    return loaded, (policy, optimizer_dict, ob_rms, epoch)

def load_from_checkpoint(policy, path, cuda):
    if not cuda:
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        state_dict = torch.load(path)
    # BC training produces names of weights with "module." in the beginning
    state_dict_renamed = {}
    for key, value in state_dict.items():
        state_dict_renamed[key.replace('module.', '')] = value
    policy.base.resnet.load_state_dict(state_dict_renamed)

def load_optimizer(optimizer, optimizer_state_dict, cuda):
    optimizer.load_state_dict(optimizer_state_dict)
    if not cuda:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

def close_envs(envs, close_envs_manually):
    envs.close()
    if close_envs_manually:
        for env in envs.venv.venv.envs:
            env.env.close()

def evaluate(policy, args, logdir, device, envs, render):
    env_config = copy.deepcopy(args)
    env_config.render = render
    num_processes = 1 if render else args.num_processes
    eval_envs = make_vec_envs(
        args.env_name, args.seed + num_processes, num_processes,
        args.gamma, logdir, args.add_timestep, device, True, env_config=env_config)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    eval_episode_rewards = []
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # while len(eval_episode_rewards) < 5:
    while len(eval_episode_rewards) < args.num_eval_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    close_envs(eval_envs, close_envs_manually=render)
    return eval_episode_rewards
