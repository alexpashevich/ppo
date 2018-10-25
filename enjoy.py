import argparse
import os

import torch

from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize, do_master_step


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-dir', default=None, required=True,
                    help='directory to load the agent from')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()

args.det = not args.non_det
device = torch.device("cuda:0" if args.cuda else "cpu")

policy, _, ob_rms, step, config_old = torch.load(os.path.join(args.load_dir, "model_current.pt"))
if hasattr(policy.base, 'resnet'):
    policy.base.resnet.eval()
policy.to(device)
config_old.render = True  # we want to render the environment

env = make_vec_envs(config_old.env_name, args.seed + 1000, 1,
                    None, None, config_old.add_timestep, device='cpu',
                    allow_early_resets=False, env_config=config_old)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
# We ignore the optimizer
print('Rendering the model after {} steps'.format(step))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

if render_func is not None:
    render_func('human')

obs = env.reset()

while True:
    with torch.no_grad():
        value, action, _, _ = policy.act(obs, None, None, deterministic=args.det)

    # Obser reward and next obs
    print('master_action = {}'.format(action))
    if not config_old.use_bcrl_setup:
        obs, reward, done, _ = env.step(action)
    else:
        obs, reward, done, _ = do_master_step(
            action, obs, config_old.timescale, policy, env)
    print('reward = {}'.format(reward.numpy()[0,0]))

    if render_func is not None:
        render_func('human')
