import numpy as np
import argparse
import gym
import mime
import matplotlib.pyplot as plt
import time
import torch
import json
import pickle as pkl
from PIL import Image

from dask.distributed import Client, LocalCluster, Pub, Sub

from bc.net.zoo import FlatPolicy
from bc.dataset import Frames, Actions
from ppo.tools import misc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--type',
        type=str,
        default='dask',
        help='Type of multiprocessing: dask or ppo')
    parser.add_argument('-np', '--num-processes', type=int, default=8)
    parser.add_argument('-nc', '--num-channels', type=int, default=3)
    parser.add_argument('-bs', '--dask-batch-size', type=int, default=8)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument(
        '-e',
        '--env-name',
        type=str,
        default='UR5-Paris-Aug-SaladRandCamEnv-v0')
    args = parser.parse_args()
    assert args.type in ('dask', 'ppo_joblib', 'ppo_dask')
    # assert args.num_processes == args.batch_size
    return args


def convert_obs(obs):
    im_keys = ['depth', 'rgb', 'mask']
    obs_im = {}
    for key, value in obs.items():
        for im_key in im_keys:
            if im_key in key:
                obs_im[im_key] = obs[key]
    obs_tensor = Frames.dic_to_tensor([obs_im], ['depth'], 1)
    return obs_tensor


def publish_obs(pub, obs, seed):
    obs_tensor = convert_obs(obs)
    pub.put((obs_tensor, seed))


def run_env(env_name, seed):
    try:
        import gym
        import mime
        env = gym.make(env_name)
        env.seed(seed)
        pub_obs = Pub('observations')
        sub_action = Sub('env{}_action'.format(int(seed)))
        obs = env.reset()
        publish_obs(pub_obs, obs, seed)
        for action in sub_action:
            obs, reward, done, info = env.step(action)
            publish_obs(pub_obs, obs, seed)
    except Exception as e:
        print(e)


def init_dask(args):
    cluster = LocalCluster(n_workers=args.num_processes)
    client = Client(cluster)
    # always define publishers first then subscribers
    pubs_action = [
        Pub('env{}_action'.format(seed)) for seed in range(args.num_processes)
    ]
    futures = client.map(run_env, [args.env_name] * args.num_processes,
                         range(args.num_processes))
    sub_obs = Sub('observations')
    return pubs_action, sub_obs


def init_policy(args):
    policy = FlatPolicy(
        archi='resnet_18_narrow32',
        mode='flat',
        num_frames=args.num_channels,
        action_space='tool_lin_ori',
        steps_action=1)
    device = torch.device('{}:0'.format(args.device))
    policy.to(args.device)
    return policy, device


def run_envs(args):
    if args.type == 'dask':
        pubs_action, sub_obs = init_dask(args)
    elif args.type == 'ppo_joblib':
        from ppo.tools.envs import make_vec_envs
        envs = make_vec_envs(
            args.env_name,
            0,
            args.num_processes,
            None,
            False,
            args.device,
            False,
            env_config=args)
    elif args.type == 'ppo_dask':
        from ppo.envs.dask import DaskEnv
        args.input_type = 'depth'
        args.max_length = None
        args.render = False
        # args.render = True
        args.augmentation = ''
        args.hrlbc_setup = True
        args.num_skills = 1
        args.timescale = 50
        args.seed = 1
        args.bc_args = dict(num_frames=args.num_channels, action_space='tool_lin_ori')
        envs = DaskEnv(args)

    policy, device = init_policy(args)
    dt_stack = []

    if args.type == 'dask':
        t0 = time.time()
        stack_obs = torch.zeros((args.dask_batch_size, args.num_channels, 224,
                                 224))
        stack_obs = stack_obs.to(device)
        stack_seed = np.zeros(args.dask_batch_size, dtype=int)
        count_obs, count_batch = 0, 0
        for obs_tensor, seed in sub_obs:
            stack_obs[count_obs] = obs_tensor
            stack_seed[count_obs] = seed
            count_obs += 1
            if count_obs >= args.dask_batch_size:
                with torch.no_grad():
                    actions = policy(stack_obs)
                for seed, action in zip(stack_seed, actions):
                    dict_action = {
                        'linear_velocity': action[:3].cpu().numpy(),
                        'angular_velocity': action[3:6].cpu().numpy(),
                        'gripper_velocity': 4
                    }
                    pubs_action[seed].put(dict_action)
                stack_obs = torch.zeros_like(stack_obs)
                count_obs = 0
                count_batch += 1
                dt_stack.append((time.time() - t0) / args.dask_batch_size)
                print('dt', np.mean(dt_stack))
                t0 = time.time()
    elif args.type == 'ppo_joblib':
        obs = envs.reset()
        t0 = time.time()
        while True:
            with torch.no_grad():
                actions = policy(obs)
            obs, reward, done, infos = envs.step(actions)
            dt_stack.append((time.time() - t0) / args.num_processes)
            print('dt', np.mean(dtstack))
            t0 = time.time()
    elif args.type == 'ppo_dask':
        from ppo.tools.misc import tensor_to_dict, dict_to_tensor
        action_keys = Actions.action_space_to_keys('tool_lin_ori')[0]
        obs = envs.reset()
        while True:
            obs, env_idxs = dict_to_tensor(obs)
            with torch.no_grad():
                actions = policy(obs)
                actions, env_idxs = tensor_to_dict(actions, env_idxs)
                for env_idx, env_action in actions.items():
                    # env_action = {'linear_velocity': np.array([-0.01, -0.01, -0.01]),
                    #               'angular_velocity': np.array([-0.01, -0.01, -0.01]),
                    #               'skill': [0]}
                    env_action = Actions.tensor_to_dict(env_action, action_keys, None)
                    env_action['skill'] = [0]
                    actions[env_idx] = env_action
            t0 = time.time()
            obs, reward, done, infos = envs.step(actions)
            dt_stack.append((time.time() - t0) / args.dask_batch_size)
            print('dt', np.mean(dt_stack), dt_stack[-5:])


def main():
    args = parse_args()
    run_envs(args)


if __name__ == '__main__':
    main()
