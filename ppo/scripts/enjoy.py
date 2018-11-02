import argparse
import os

import torch

from ppo.tools.envs import VecPyTorch, make_vec_envs
from ppo.tools.utils import get_render_func, get_vec_normalize, do_master_step, get_device


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--load-dir', default=None, required=True,
                        help='directory to load the agent from')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--non-det', action='store_true', default=False,
                        help='whether to use a non-deterministic policy')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which device to run the experiments on: cuda or cpu')
    parser.add_argument('--num-whiles', type=int, default=420,
                        help='number of episodes to render')
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='whether to render the environment')
    args = parser.parse_args()
    args.det = not args.non_det
    args.render = not args.no_render
    return args

def main():
    args = get_args()
    device = get_device(args.device)

    # We need to use the statistics for normalization from the training, we ignore the optimizer
    policy, _, ob_rms, step, config_old = torch.load(os.path.join(args.load_dir, "model_current.pt"))
    if hasattr(policy.base, 'resnet'):
        # set the batch norm to the eval mode
        policy.base.resnet.eval()
    policy.to(device)
    config_old.render = args.render

    env = make_vec_envs(config_old.env_name, args.seed + 1000, 1,
                        None, config_old.add_timestep, device,
                        allow_early_resets=False, env_config=config_old)

    print('Rendering the model after {} steps'.format(step))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    obs = env.reset()

    for _ in range(args.num_whiles):
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

if __name__ == '__main__':
    main()
