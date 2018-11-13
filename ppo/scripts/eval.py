import argparse
import os
import time
import torch
import glob

# import ppo.tools.utils as utils
from ppo.tools.utils import get_device, evaluate, seed_torch
from ppo.tools.envs import make_vec_envs
from ppo.tools.log import init_writers, log_eval
from ppo.tools.gifs import save as save_gifs

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--expdir', required=True,
                        help='$exp/$seed/$timestamp directory of the experiment to evaluate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which device to run the experiments on: cuda or cpu')
    parser.add_argument('--save-gifs', action='store_true', default=False,
                        help='whether to save the gifs of the evaluation environments')
    parser.add_argument('--num-episodes', type=int, default=32,
                        help='number of episodes to use in evaluation')
    parser.add_argument('--max-length', type=int, default=600,
                        help='episodes max length')
    # parser.add_argument('--no-render', action='store_true', default=False,
    #                     help='whether to render the environment')
    parser.add_argument('--logdir', type=str, default=None, help='not used')
    parser.add_argument('--timestamp', type=str, default=None, help='not used')
    args = parser.parse_args()
    return args

def run_evaluation(load_path, args, device, epoch, eval_envs):
    if str(device) == 'cpu':
        load_tuple = torch.load(load_path, map_location=lambda storage, loc: storage)
    else:
        load_tuple = torch.load(load_path)
    policy, _, ob_rms, step, args_old = load_tuple
    assert args_old.eval_offline, 'evaluation was already done'
    args_old.eval_max_length_factor = 1
    args_old.max_length = args.max_length
    args_old.num_eval_episodes = args.num_episodes
    args_old.seed = args.seed
    args_old.save_gifs = args.save_gifs
    if hasattr(policy.base, 'resnet'):
        # set the batch norm to the eval mode
        policy.base.resnet.eval()
    policy.to(device)

    # implement if necessary later
    env_render = None
    # env_render_train, env_render_eval = create_render_env(args, device) if render else (None, None)

    eval_envs, stats, gifs = evaluate(policy=policy,
                                      args_train=args_old,
                                      device=device,
                                      train_envs_or_ob_rms=ob_rms,
                                      eval_envs=eval_envs,
                                      env_render=env_render)
    num_env_steps = (epoch + 1) * args_old.num_processes * args_old.num_frames_per_update
    return stats, gifs, num_env_steps, eval_envs

def main():
    args = get_args()
    device = get_device(args.device)
    seed_torch(args)
    print('Evaluating the experiments on {}'.format(device))

    #assert not os.path.exists(os.path.join(args.expdir, 'gifs')), 'gifs dir already exist, seems'
    init_writers(None, os.path.join(args.expdir, 'eval'))
    eval_envs = None
    load_paths = glob.glob(os.path.join(args.expdir, 'model_eval_*.pt'))
    load_paths_sorted = sorted(load_paths, key=lambda s: int(s.split('_')[-1].replace('.pt', '')))
    # for load_path in reversed(load_paths_sorted):
    for load_path in load_paths_sorted:
        epoch = int(load_path.split('model_eval_')[-1].replace('.pt', ''))
        gifs_path_epoch = os.path.join(args.expdir, 'gifs', 'epoch{}'.format(epoch))
        if os.path.exists(gifs_path_epoch):
            print('Gif directory for epoch {} already exists, skipping the epoch...')
            continue
        start = time.time()
        stats, gifs, num_env_steps, eval_envs = run_evaluation(load_path, args, device, epoch, eval_envs)
        print("Evaluation of epoch {} ({} steps) took {:.1f} seconds".format(
            epoch, num_env_steps, time.time() - start))
        log_eval(num_env_steps, stats)
        save_gifs(args.expdir, gifs, epoch)


if __name__ == "__main__":
    main()
