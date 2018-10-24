import numpy as np
import json
import argparse
import torch
from gym.spaces import Discrete

from model import MasterPolicy
from utils import load_from_checkpoint, do_master_step
from envs import make_vec_envs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to render the evaluation')
    parser.add_argument('--pudb', action='store_true', default=False,
                        help='whether to stop at the breakpoint for manual actions switching')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether to run the policy on cuda')
    parser.add_argument('--no-report-failures', action='store_true', default=False,
                        help='whether to report when the environment is not done')
    parser.add_argument('--env-name', default='UR5-BowlEnv-v0',
                        help='environment to train on (default: UR5-BowlEnv-v0)')
    parser.add_argument('--timescale', type=int, default=25,
                        help='master timescale')
    parser.add_argument('--num-skills', type=int, default=4,
                        help='number of skills')
    parser.add_argument('--action-sequence', type=json.loads, default=[0, 0, 1, 2, 3],
                        help='number of skills')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='number of episodes to run')
    # num-processes > 1 does not work at the moment
    parser.add_argument('--num-processes', type=int, default=1,
                        help='number of processes to run')
    # BCRL hieararchy
    parser.add_argument('--use-bcrl-setup', action='store_true', default=False,
                        help='test the pretrained skills')
    parser.add_argument('--dim-skill-action', type=int, default=5,
                        help='dimensionality of a skill action')
    parser.add_argument('--num-skill-action-pred', type=int, default=1,
                        help='dimensionality of a skill action')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='if specified, load the networks weights from the file')
    args = parser.parse_args()
    return args

def _perform_actions(action_sequence, envs, policy, obs, args):
    reward_sum = np.array([0] * args.num_processes, dtype=np.float)
    for action in action_sequence:
        print('master action = {}'.format(action))
        action = np.array([action])
        if not args.use_bcrl_setup:
            action = torch.tensor(action)[None]
            _, reward, done, info = envs.step(action)
        else:
            obs, reward, done, info = do_master_step(
                action, obs, args.timescale, policy, envs)
        reward_sum += reward[:, 0].numpy()
    return reward_sum, done, info

def main():
    args = get_args()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(
        args.env_name, 1, args.num_processes, 0.99, None, False, device, True, env_config=args)
    obs = envs.reset()
    policy = None
    if args.num_episodes % args.num_processes != 0:
        print('Warning: num_episodes can not be divided by num_processes: I may run less episodes')
    if args.render:
        assert args.num_processes == 1, 'can not render more than 1 process'

    if args.use_bcrl_setup:
        assert args.checkpoint_path, 'does not make sense to test the random network'
        assert 'Cam' in args.env_name, 'bcrl setup works only from the depth input'
        print('Using BCRL setup, loading the skills from {}'.format(args.checkpoint_path))
        action_space = Discrete(args.num_skills)
        policy = MasterPolicy(envs.observation_space.shape, action_space, base_kwargs=vars(args))
        load_from_checkpoint(policy, args.checkpoint_path, args.cuda)
        policy.to(device)

    episodes_success = []
    perform_actions = lambda seq: _perform_actions(seq, envs, policy, obs, args)
    if args.pudb:
        # you can call, e.g. perform_actions([0, 0, 1, 2, 3]) in the terminal
        import pudb; pudb.set_trace()
    for epoch in range(args.num_episodes // args.num_processes):
        reward_sum, done, info = perform_actions(args.action_sequence)
        success = np.count_nonzero(reward_sum > 0)
        episodes_success.append(success)
        if not done.all() and args.no_report_failures is False:
            # one of the envs is not done after the sequence of actions
            print('WARNING [epoch {}]: done = {}, info = {}'.format(epoch, done, info))
            envs.reset()
    envs.close()
    print('Success rate of {} episodes is {}'.format(
        args.num_episodes // args.num_processes * args.num_processes,
        np.mean(episodes_success)))


if __name__ == '__main__':
    main()
