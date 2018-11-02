import numpy as np
import json
import argparse
import torch
from gym.spaces import Discrete

from ppo.parts.model import MasterPolicy
from ppo.tools.utils import load_from_checkpoint, do_master_step, set_up_training, get_device
from ppo.tools.envs import make_vec_envs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False,
                        help='whether to render the evaluation')
    parser.add_argument('--pudb', action='store_true', default=False,
                        help='whether to stop at the breakpoint for manual actions switching')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which device to run the experiments on: cuda or cpu')
    parser.add_argument('--input-type', '-i', type=str, default='rgbd',
                        help='type of input for the conv nets')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed to use')
    parser.add_argument('--no-report-failures', action='store_true', default=False,
                        help='whether to report when the environment is not done')
    parser.add_argument('--env-name', default='UR5-BowlCamEnv-v0',
                        help='environment to train on (default: UR5-BowlEnv-v0)')
    parser.add_argument('--timescale', type=int, default=60,
                        help='master timescale')
    parser.add_argument('--num-skills', type=int, default=4,
                        help='number of skills')
    parser.add_argument('--action-sequence', type=json.loads, default=[0, 0, 0, 1, 2, 3, 3, 3],
                        help='number of skills')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='number of episodes to run')
    parser.add_argument('--num-processes', '-np', type=int, default=1,
                        help='number of processes to run')
    # BCRL hieararchy
    parser.add_argument('--no-use-bcrl-setup', action='store_true', default=False,
                        help='test the pretrained skills')
    parser.add_argument('--dim-skill-action', '-da', type=int, default=8,
                        help='dimensionality of a skill action')
    parser.add_argument('--num-skill-action-pred', '-nap', type=int, default=4,
                        help='dimensionality of a skill action')
    parser.add_argument('--archi', type=str, default='resnet18',
                        help='which architecture to use (from bc.net.architectures.resnet)')
    parser.add_argument('--checkpoint-path', '-cp', type=str, default=None,
                        help='if specified, load the networks weights from the file')
    args = parser.parse_args()
    args.use_bcrl_setup = not args.no_use_bcrl_setup
    return args

def _perform_actions(action_sequence, envs, policy, obs, args):
    reward_glob = np.array([0] * args.num_processes, dtype=np.float)
    done_glob = np.array([False] * args.num_processes, dtype=np.bool)
    info_glob = [{}] * args.num_processes
    for action in action_sequence:
        action = np.array([action for _ in range(args.num_processes)])
        if not args.use_bcrl_setup:
            action = torch.tensor(action)[None]
            _, reward_step, done_step, info_step = envs.step(action)
        else:
            obs, reward_step, done_step, info_step = do_master_step(
                action, obs, args.timescale, policy, envs)
        for env_id, done_before_step in enumerate(done_glob):
            if not done_before_step:
                reward_glob[env_id] += reward_step[env_id, 0].numpy()
                info_glob[env_id] = info_step[env_id]
            else:
                # else the environment was done before and a new init config is running which we ignore
                print('env {} is already done'.format(env_id))
        done_glob = np.logical_or(done_glob, done_step)
        if done_glob.all():
            break
    return reward_glob, done_glob, info_glob


def main():
    args = get_args()
    device = get_device(args.device)
    set_up_training(args)
    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, 0.99, False, device, True, env_config=args)
    obs = envs.reset()
    policy = None
    if args.num_episodes % args.num_processes != 0:
        print('Warning: num_episodes can not be divided by num_processes: I may run less episodes')
    if args.render:
        assert args.num_processes == 1, 'can not render more than 1 process'

    if args.use_bcrl_setup:
        assert args.checkpoint_path, 'does not make sense to test the random network'
        assert 'Cam' in args.env_name, 'bcrl setup works only from the depth input'
        print('Using the BCRL setup')
        action_space = Discrete(args.num_skills)
        policy = MasterPolicy(envs.observation_space.shape, action_space, base_kwargs=vars(args))
        load_from_checkpoint(policy, args.checkpoint_path, device)
        policy.to(device)
    if policy and hasattr(policy.base, 'resnet'):
        # set the batch norm to the eval mode
        policy.base.resnet.eval()

    episodes_success = []
    perform_actions = lambda seq: _perform_actions(seq, envs, policy, obs, args)
    if args.pudb:
        # you can call, e.g. perform_actions([0, 0, 1, 2, 3]) in the terminal
        import pudb; pudb.set_trace()
    for epoch in range(args.num_episodes // args.num_processes):
        reward_sum, done, info = perform_actions(args.action_sequence)
        success = np.count_nonzero(reward_sum > 0)
        print('epoch success = {}'.format(success))
        episodes_success.append(success)
        # TODO: do we need to reset here? wtf the success rate is decreasing over time?
        envs.reset()
        # if not done.all() and args.no_report_failures is False:
        #     # one of the envs is not done after the sequence of actions
        #     print('WARNING [epoch {}]: done = {}, info = {}'.format(epoch, done, info))
    envs.close()
    num_episodes_done = args.num_episodes // args.num_processes * args.num_processes
    print('Success rate of {} episodes is {}'.format(
        num_episodes_done,
        np.sum(episodes_success) / num_episodes_done))


if __name__ == '__main__':
    main()
