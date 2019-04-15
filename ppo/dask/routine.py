import gym
import torch
import itertools
import numpy as np

from collections import OrderedDict, deque
from dask.distributed import Pub, Sub

from bc.dataset import Frames, Actions
from ppo.tools import misc


class AsynchMimeEnv:
    def __init__(self, args):
        self.parse_args(args)

        # create variables
        self.env = self.create_env(args['seed'])
        self.pub_out = Pub('observations')
        self.sub_in = Sub('env{}_input'.format(int(self.env_idx)))
        self.frames_stack = deque(maxlen=args['num_frames_stacked'])
        self.reset_vars()

        # start the environment loop
        self.env_loop()

    def parse_args(self, args):
        # parse the args
        self.env_idx = args['env_idx']
        self.env_name = args['env_name']
        self.max_length = args['max_length']
        self.render = args['render']
        # I use -1 because I count the griper as 2 vals
        self.action_keys = Actions.action_space_to_keys(
            args['robot_action_space'], args['dim_skill_action'] - 1)
        if args['input_type'] == 'depth':
            self.channels = ('depth',)
        elif args['input_type'] == 'rgbd':
            self.channels = ('depth', 'rgb')
        else:
            raise NotImplementedError('Unknown input type = {}'.format(args['input_type']))
        self.augmentation = args['augmentation']
        self.hrlbc_setup = args['hrlbc_setup']
        num_skills = args['num_skills']
        # timescales for skills (hrlbc setup only)
        self.skills_timescales = args['timescale']
        if self.hrlbc_setup:
            assert self.skills_timescales is not None
        if isinstance(self.skills_timescales, int):
            skills_timescales = {}
            for skill in range(num_skills):
                skills_timescales[skill] = self.skills_timescales
            self.skills_timescales = skills_timescales

    def env_loop(self):
        for input_ in self.sub_in:
            self.step_counter += 1
            self.step_counter_after_new_action += 1

            if input_['function'] == 'reset':
                obs = self.env.reset()
                self.reset_vars()
                self.publish_obs(obs_dict={'observation': obs})
            elif input_['function'] == 'step':
                action_applied = self.get_action_applied(input_['action'])
                obs, reward, done, info = self.env.step(action_applied)
                info = self.update_info(info)
                self.publish_obs(
                    obs_dict={'observation': obs, 'reward': reward, 'done': done, 'info': info})
                if done:
                    self.reset_vars()
            else:
                raise NotImplementedError('function {} is not implemented'.format(
                    input_['function']))

    def create_env(self, seed):
        env = gym.make(self.env_name)
        env.seed(self.env_idx + seed)
        if hasattr(env.unwrapped.scene, 'set_rl_mode'):
            env.unwrapped.scene.set_rl_mode()
        else:
            print('WARNING: the scene does not have set_rl_mode function')
        if self.max_length is not None:
            env._max_episode_steps = self.max_length
        if self.render and self.env_idx == 0:
            env.unwrapped.scene.renders(True)
        return env

    def reset_vars(self):
        self.step_counter = 0
        self.step_counter_after_new_action = 0
        self.frames_stack.clear()
        self.prev_script = None
        self.need_master_action = True

    def update_info(self, info):
        if len(info['failure_message']):
            print('env {} failure {}'.format(self.env_idx, info['failure_message']))
        info['length'] = self.step_counter
        info['need_master_action'] = self.need_master_action
        info['length_after_new_action'] = self.step_counter_after_new_action
        return info

    def publish_obs(self, obs_dict):
        obs_tensor = self.convert_obs(obs_dict['observation'])
        obs_dict['observation'] = obs_tensor
        self.pub_out.put((obs_dict, self.env_idx))

    def convert_obs(self, obs_dict):
        if 'Cam' not in self.env_name:
            # TODO: this was not debugged at all
            misc.pudb()
            observation = np.array([])
            obs_sorted = OrderedDict(sorted(obs_dict.items(), key=lambda t: t[0]))
            for obs_key, obs_value in obs_sorted.items():
                if obs_key != 'skill':
                    if isinstance(obs_value, (int, float)):
                        obs_value = [obs_value]
                    elif isinstance(obs_value, np.ndarray):
                        obs_value = obs_value.flatten()
                    elif isinstance(obs_value, list) and isinstance(obs_value[0], np.ndarray):
                        obs_value = np.concatenate(obs_value)
                    observation = np.concatenate((observation, obs_value))
            obs_tensor = torch.tensor(observation)
        else:
            im_keys = ['depth', 'rgb', 'mask']
            obs_im = {}
            for key, value in obs_dict.items():
                for im_key in im_keys:
                    if im_key in key:
                        obs_im[im_key] = obs_dict[key]
            self.frames_stack.append(obs_im)
            while len(self.frames_stack) < self.frames_stack.maxlen:
                self.frames_stack.append(obs_im)
            obs_tensor = Frames.dic_to_tensor(
                self.frames_stack,
                self.channels,
                Frames.sum_channels(self.channels),
                augmentation_str=self.augmentation)
        return obs_tensor

    def get_action_applied(self, action):
        if self.hrlbc_setup:
            skill = action.pop('skill')[0]
            if self.step_counter_after_new_action >= self.skills_timescales[skill]:
                print('env {} needs a new master action (skill = {}, ts = {})'.format(
                    self.env_idx, skill, self.step_counter))
                self.need_master_action = True
                self.step_counter_after_new_action = 0
            else:
                self.need_master_action = False
            return action

        if self.prev_script != action:
            self.prev_script = action
            self.prev_action_chain = self.env.unwrapped.scene.script_subtask(action)
        action_chain = itertools.chain(*self.prev_action_chain)
        action_applied = Actions.get_dict_null_action(self.action_keys)
        action_update = next(action_chain, None)
        if action_update is None:
            print('env {} needs a new master action'.format(self.env_idx))
            self.need_master_action = True
            self.step_counter_after_new_action = 0
        else:
            self.need_master_action = False
            action_applied.update(Actions.filter_action(action_update, self.action_keys))
        return action_applied
