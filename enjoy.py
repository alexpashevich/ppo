import argparse
import os

import torch

from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='UR5-BowlEnv-v0',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--timescale', type=int, default=25,
                    help='master timescale')
args = parser.parse_args()

args.det = not args.non_det
# BowlEnv specific
args.num_skills = 4
args.render = True

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                    None, None, args.add_timestep, device='cpu',
                    allow_early_resets=False, env_config=args)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
# We ignore the optimizer
actor_critic, _, ob_rms, step = torch.load(os.path.join(args.load_dir, "model_current.pt"))
print('Rendering the model after {} steps'.format(step))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    print('reward = {}'.format(reward.numpy()[0,0]))

    masks.fill_(0.0 if done else 1.0)

    if render_func is not None:
        render_func('human')
