import argparse
import os
import torch

from ppo.tools.envs import make_vec_envs
from ppo.tools.misc import get_vec_normalize, get_device
from ppo.scripts.utils import do_master_step


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--load-path', default=None, required=True,
                        help='directory or checkpoint path to load the agent from')
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
    parser.add_argument('--same-path-for-json', action='store_true', default=False,
                        help='whether to load info.json from the load_path or from the old config path')
    args = parser.parse_args()
    args.det = not args.non_det
    args.render = not args.no_render
    return args

def main():
    args = get_args()
    device = get_device(args.device)

    # We need to use the statistics for normalization from the training, we ignore the optimizer
    load_path = args.load_path if '.pt' in args.load_path else os.path.join(args.load_path,
                                                                            "model_current.pt")
    policy, _, ob_rms, step, config_old = torch.load(load_path)
    if args.same_path_for_json and config_old.checkpoint_path is not None:
        load_dir = args.load_path if '.pt' not in args.load_path else os.path.dirname(args.load_path)
        checkpoint_path_new = os.path.join(load_dir, 'info.json')
        assert os.path.exists(checkpoint_path_new), 'You should have info.json in the same folder'
        config_old.checkpoint_path = checkpoint_path_new
    if hasattr(policy.base, 'resnet'):
        # set the batch norm to the eval mode
        policy.base.resnet.eval()
    policy.to(device)
    config_old.render = args.render

    env = make_vec_envs(config_old.env_name, args.seed + 1000, 1,
                        None, config_old.add_timestep, device,
                        False, env_config=config_old)

    print('Rendering the model after {} steps'.format(step))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    obs = env.reset()

    for _ in range(args.num_whiles):
        with torch.no_grad():
            value, action, _, _ = policy.act(obs, None, None, deterministic=args.det)

        # Observe reward and next obs
        obs, reward, done, _ = do_master_step(
            action, obs, config_old.timescale, policy, env, config_old.hrlbc_setup)
        print('reward = {}'.format(reward.numpy()[0,0]))

if __name__ == '__main__':
    main()
