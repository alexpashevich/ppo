import gym
import mime
import torch
import os
import json
import itertools
import faulthandler
import numpy as np
from bc.utils.videos import write_video

from collections import OrderedDict
from dask.distributed import Pub, Sub

from bc.dataset import Frames, Actions
from ppo.tools import misc


class MimeEnv:
    def __init__(self, args, obs_running_stats=None):
        self.parse_args(args)

        # create variables
        self.env = self.create_env(args['seed'])
        self.pub_out = Pub('observations')
        self.sub_in = Sub('env{}_input'.format(int(self.env_idx)))
        self.step_counter = 0
        self.reset_env(reset_mime=False)

        # clean the dask pipes
        self.pub_out.put(self.env_idx)
        # make the env to print all the logs
        faulthandler.enable()

        # start the environment loop
        self.env_loop()

    def parse_args(self, args):
        # parse the args
        self.env_idx = args['env_idx']
        self.env_name = args['env_name']
        self.max_length = args['max_length']
        self.render = args['render'] and self.env_idx == 0
        self.action_keys = Actions.action_space_to_keys(args['bc_args']['action_space'])[0]
        if args['input_type'] == 'depth':
            self.channels = ('depth',)
        elif args['input_type'] == 'rgbd':
            self.channels = ('depth', 'rgb')
        else:
            raise NotImplementedError('Unknown input type = {}'.format(args['input_type']))
        self.augmentation = args['augmentation']
        self.hrlbc_setup = args['hrlbc_setup']
        self.check_skills_silency = args['check_skills_silency']
        num_skills = args['num_skills']
        # timescales for skills (hrlbc setup only)
        self.skills_timescales = args['timescale']
        if self.hrlbc_setup:
            assert self.skills_timescales is not None
        if isinstance(self.skills_timescales, int):
            skills_timescales = {}
            for skill in range(num_skills):
                skills_timescales[str(skill)] = self.skills_timescales
            self.skills_timescales = skills_timescales
        # gifs writing
        self.gifdir = None
        if 'gifdir' in args:
            self.gifdir = os.path.join(args['gifdir'], '{:02d}'.format(self.env_idx))
            self.gif_counter = 0
            if self.gifdir:
                self.obs_history = {}

    def env_loop(self):
        for input_ in self.sub_in:
            self.step_counter += 1
            self.step_counter_after_new_action += 1

            if input_['function'] == 'reset':
                obs = self.reset_env()
                self.publish_obs(obs_dict={'observation': obs})
            elif input_['function'] == 'reset_after_crash':
                obs = self.reset_env()
                self.publish_obs(
                    obs_dict={'observation': obs, 'reward': 0, 'done': True,
                              'info': self.update_info(
                                  {'success': False, 'failure_message': 'Env crashed'})})
            elif input_['function'] == 'step':
                action_applied = self.get_action_applied(input_['action'])
                obs, reward, done, info = self.env.step(action_applied)
                info = self.update_info(info)
                if done:
                    obs = self.reset_env(error_message=info['failure_message'])
                self.publish_obs(
                    obs_dict={'observation': obs, 'reward': reward, 'done': done, 'info': info})
            else:
                raise NotImplementedError('function {} is not implemented'.format(
                    input_['function']))

    def create_env(self, seed):
        env = gym.make(self.env_name)
        env.seed(self.env_idx + seed)
        if hasattr(env.unwrapped.scene, 'set_rl_mode'):
            env.unwrapped.scene.set_rl_mode(hrlbc=self.hrlbc_setup)
        else:
            raise NotImplementedError
        if self.max_length is not None:
            env._max_episode_steps = self.max_length
        if self.render:
            env.unwrapped.scene.renders(True)
        return env

    def reset_env(self, reset_mime=True, error_message=''):
        step_counter_cached = self.step_counter
        self.step_counter = 0
        self.step_counter_after_new_action = 0
        self.prev_script = None
        self.need_master_action = True
        if self.gifdir:
            for obs_key, obs_list in self.obs_history.items():
                if obs_key != 'skills':
                    gif_name = os.path.join(
                        self.gifdir, 'obs_{}_{}.mp4'.format(self.gif_counter, obs_key))
                    write_video(obs_list, gif_name)
                else:
                    json_name = os.path.join(self.gifdir, 'obs_skills_{}.json'.format(self.gif_counter))
                    with open(json_name, 'w') as json_file:
                        json.dump(obs_list, json_file)
            if len(self.obs_history) > 0:
                self.gif_counter += 1
                self.obs_history = {}
        if reset_mime:
            obs = self.env.reset()
            print('env {:02d} is reset after {} timesteps: {}'.format(
                self.env_idx, step_counter_cached - 1, error_message))
            return obs

    def update_info(self, info):
        info['length'] = self.step_counter
        info['need_master_action'] = self.need_master_action
        info['length_after_new_action'] = self.step_counter_after_new_action
        if self.need_master_action:
            self.step_counter_after_new_action = 0
        return info

    def publish_obs(self, obs_dict):
        obs_tensor = self.convert_obs(obs_dict['observation'])
        obs_dict['observation'] = obs_tensor
        self.pub_out.put((obs_dict, self.env_idx))

    def convert_obs(self, obs_dict):
        if 'Cam' not in self.env_name:
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
            obs_tensor = torch.tensor(observation).float()
        else:
            im_keys = ['depth', 'rgb', 'mask']
            obs_im = {}
            for key, value in obs_dict.items():
                for im_key in im_keys:
                    if im_key in key:
                        obs_im[im_key] = obs_dict[key]
            obs_tensor = Frames.dict_to_tensor(
                [obs_im],
                self.channels,
                Frames.sum_channels(self.channels),
                augmentation_str=self.augmentation)
            if self.gifdir:
                if 'orig' in self.obs_history:
                    self.obs_history['orig'].append(obs_im['depth'])
                else:
                    self.obs_history['orig'] = [obs_im['depth']]
                obs_tensor_denormalized = (obs_tensor[0].numpy() + 1) / 2 * 255
                if 'aug' in self.obs_history:
                    self.obs_history['aug'].append(obs_tensor_denormalized)
                else:
                    self.obs_history['aug'] = [obs_tensor_denormalized]
        return obs_tensor

    def get_action_applied(self, action):
        skill = action.pop('skill')[0]
        if self.step_counter_after_new_action == 1:
            if 'skills' in self.obs_history:
                self.obs_history['skills'].append(int(skill))
            else:
                self.obs_history['skills'] = [int(skill)]
            if self.render:
                print('env {:02d} got a new master action = {} (ts = {})'.format(
                    self.env_idx, skill, self.step_counter))
        if self.hrlbc_setup:
            if self.step_counter_after_new_action >= self.skills_timescales[str(skill)]:
                if self.render:
                    print('env {:02d} needs a new master action (ts = {})'.format(
                        self.env_idx, self.step_counter))
                self.need_master_action = True
            else:
                self.need_master_action = False
            if self.check_skills_silency and self.step_counter_after_new_action == 1:
                if self.env.unwrapped.scene.skill_should_be_silent(skill):
                    if self.render:
                        print('env {:02d} skips the action {} (ts = {})'.format(
                            self.env_idx, skill, self.step_counter))
                    action = dict(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
                    self.need_master_action = True
            return action
        else:
            return self.get_script_action(skill)

    def get_script_action(self, skill):
        if self.prev_script != skill:
            self.prev_script = skill
            self.prev_action_chain = self.env.unwrapped.scene.script_subtask(skill)
        action_chain = itertools.chain(*self.prev_action_chain)
        action_applied = Actions.get_dict_null_action(self.action_keys)
        action_update = next(action_chain, None)
        if action_update is None:
            if self.render:
                print('env {:02d} needs a new master action (ts = {})'.format(
                    self.env_idx, self.step_counter))
            self.need_master_action = True
        else:
            self.need_master_action = False
            action_applied.update(Actions.filter_action(action_update, self.action_keys))
        if self.skills_timescales is not None:
            skill_timescale = self.skills_timescales[str(skill)]
            self.need_master_action = self.step_counter_after_new_action >= skill_timescale
        return action_applied
