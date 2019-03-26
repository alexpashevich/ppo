import gym
import itertools
import json
import os
import numpy as np
import pickle as pkl

from collections import OrderedDict, deque
from gym.spaces import Box, Discrete
from io import BytesIO

from mime.agent.agent import Agent
from bc.dataset import Frames


SUPPORTED_MIME_ENVS = 'Bowl', 'Salad'

class MiMEEnv(object):
    def __init__(self, env_name, config, id=0):
        assert any([env_prefix in env_name for env_prefix in SUPPORTED_MIME_ENVS])

        self.env_name = env_name
        self.env = gym.make(env_name)
        if 'max_length' in vars(config) and config.max_length is not None:
            self.env._max_episode_steps = config.max_length
        self.num_skills = vars(config).get('num_skills', 4)
        self.timescale = vars(config).get('timescale', 25)
        self._render = vars(config).get('render', False) and id == 0
        self._id = id
        self.hrlbc_setup = vars(config).get('hrlbc_setup', False)
        self.observation_type = vars(config).get('input_type', 'depth')
        self.augmentation = vars(config).get('augmentation', False)
        # some copypasting
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        # activate rendering
        if self._render:
            scene = self.env.unwrapped.scene
            scene.renders(True)
        # bcrl related stuff
        self.dim_skill_action = vars(config).get('dim_skill_action', 8)

        self.action_keys = list(self.env.action_space.spaces.keys())
        if vars(config).get('checkpoint_path', None) is not None:
            # load action statistics for the denormalization
            self.action_mean, self.action_std = self._load_action_stats(
                config.checkpoint_path)
        else:
            self.action_mean, self.action_std = None, None
        self.frames_stack = deque(maxlen=3)
        self.compress_frames = True

    def _load_action_stats(self, checkpoint_path):
        checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
        infos_path = os.path.join(checkpoint_dir, 'info.json')
        stats = json.load(open(infos_path, 'r'))['statistics']
        # get network dataset statistics
        mean, std = {}, {}
        for action_key in self.action_keys:
            mean[action_key], std[action_key] = stats['action'][action_key]
        return mean, std

    @property
    def observation_space(self):
        if 'Cam' in self.env_name:
            if self.observation_type == 'depth':
                observation_dim = 1 * 3
            elif self.observation_type == 'rgbd':
                observation_dim = 4 * 3
            else:
                raise NotImplementedError
            return Box(-np.inf, np.inf, (observation_dim, 224, 224), dtype=np.float)
        elif 'Bowl' in self.env_name:
            return Box(-np.inf, np.inf, (19,), dtype=np.float)
        elif 'Salad' in self.env_name:
            num_cups = self.env.unwrapped.scene._num_cups
            num_drops = self.env.unwrapped.scene._num_drops
            num_features = 32 + 10 * num_cups + 3 * num_drops * num_cups
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)

    @property
    def action_space(self):
        if not self.hrlbc_setup:
            return Discrete(self.num_skills)
        else:
            return Box(-np.inf, np.inf, (self.dim_skill_action,), dtype=np.float)

    def _process_obs(self, obs_dict):
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
        else:
            if self.observation_type == 'depth':
                channels = ('depth',)
            elif self.observation_type == 'rgbd':
                channels = ('depth', 'rgb')
            obs_im = {}
            for channel in channels:
                obs_im[channel] = obs_dict['{}0'.format(channel)].copy()
            if 'mask0' in obs_dict:
                obs_im['mask'] = obs_dict['mask0'].copy()
            self.frames_stack.append(obs_im)
            while len(self.frames_stack) < 3:
                # if this is the first observation after reset
                self.frames_stack.append(obs_im)
            observation = Frames.dic_to_tensor(
                self.frames_stack, channels, Frames.sum_channels(channels),
                augmentation_str=self.augmentation)
        return observation

    def _get_null_action_dict(self):
        action_dict = {}
        for action_key in self.action_keys:
            if action_key == 'grip_velocity':
                action_dict[action_key] = 0
            elif action_key in ('linear_velocity', 'angular_velocity'):
                action_dict[action_key] = [0] * 3
            elif action_key == 'joint_velocity':
                action_dict[action_key] = [0] * 6
            else:
                raise ValueError('Unknown action_key = {}'.format(action_key))
        return action_dict

    def _action_numpy_to_dict(self, action):
        # we multiply the actions by std and shift by mean (except for the gripper velocity)
        action_dict = self._get_null_action_dict()
        for action_key in self.action_keys:
            if action_key != 'grip_velocity':
                if action_key == 'joint_velocity':
                    action_value = action[2:8]
                elif action_key == 'linear_velocity':
                    action_value = action[2:5]
                elif action_key == 'angular_velocity':
                    action_value = action[5:8]
                action_value *= self.action_std[action_key]
                action_value += self.action_mean[action_key]
                action_dict[action_key] = action_value
            else:
                action_value = -1 + 2 * (action[0] > action[1])
                action_dict['grip_velocity'] = action_value * 2
        return action_dict

    def _filter_action(self, action):
        # the scripts of the mime env seems to always return both IK and joints velocities
        # we filter the action depending on the action_space of the mime environment
        action_filtered = {}
        for action_key, action_value in action.items():
            if action_key in self.env.action_space.spaces.keys():
                action_filtered[action_key] = action_value
        return action_filtered

    def _get_action_applied(self, action):
        # get the action either from numpy or from a script
        if self.hrlbc_setup:
            action_applied = self._action_numpy_to_dict(action)
        else:
            if self._prev_script != action:
                if self._render:
                    print('got a new script for env {}'.format(self._id))
                self._prev_script = action
                self._prev_action_chain = self.env.unwrapped.scene.script_subtask(action)
            action_chain = itertools.chain(*self._prev_action_chain)
            action_applied = self._get_null_action_dict()
            action_update = self._filter_action(next(action_chain, action_applied))
            action_applied.update(action_update)
        return action_applied

    def step(self, action):
        action_applied = self._get_action_applied(action)
        obs, reward, done, info = self.env.step(action_applied)
        observation = self._process_obs(obs)
        if len(info['failure_message']):
            print('env {} failure {}'.format(self._id, info['failure_message']))
        if self.compress_frames:
            buffer = BytesIO()
            pkl.dump(observation.numpy(), buffer)
            observation = buffer.getvalue()
        return observation, reward, done, info

    def reset(self):
        self.frames_stack.clear()
        self._prev_script = None
        print('env {} is reset'.format(self._id))
        if self._render:
            print('env is reset')
        obs = self.env.reset()
        observation = self._process_obs(obs)
        if self.compress_frames:
            buffer = BytesIO()
            pkl.dump(observation.numpy(), buffer)
            observation = buffer.getvalue()
        return observation

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode):
        self.env.render(mode)

    def close(self):
        self.env.close()
