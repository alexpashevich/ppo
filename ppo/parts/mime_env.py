import gym
import itertools
import numpy as np
import json
import os

from collections import OrderedDict
from gym.spaces import Box, Discrete
from copy import deepcopy
from torchvision import transforms

from mime.agent.agent import Agent


SUPPORTED_MIME_ENVS = 'UR5-BowlEnv-v0', 'UR5-BowlCamEnv-v0', 'UR5-SaladEnv-v0', 'UR5-SaladCamEnv-v0'

class MiMEEnv(object):
    def __init__(self, env_name, config, id=0):
        assert env_name in SUPPORTED_MIME_ENVS

        self.env_name = env_name
        self.env = gym.make(env_name)
        if 'max_length' in vars(config) and config.max_length is not None:
            self.env._max_episode_steps = config.max_length
        self.num_skills = vars(config).get('num_skills', 4)
        self.timescale = vars(config).get('timescale', 25)
        self._render = vars(config).get('render', False) and id == 0
        self._skip_unused_obs = vars(config).get('skip_unused_obs', False)
        self._id = id
        self.use_bcrl_setup = vars(config).get('use_bcrl_setup', False)
        # pure PPO baseline
        self.use_direct_actions = vars(config).get('use_direct_actions', False)
        if self.use_direct_actions:
            # ugly quick fix
            self.use_bcrl_setup = True
        self.num_frames = 1 if self.use_bcrl_setup else 3
        self.observation_type = vars(config).get('input_type', 'depth')
        # some copypasting
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        # activate rendering
        if self._render:
            scene = self.env.unwrapped.scene
            scene.renders(True)
        # bcrl related stuff
        self.dim_skill_action = vars(config).get('dim_skill_action', 5)

        self.action_keys = list(self.env.action_space.spaces.keys())
        if vars(config).get('checkpoint_path', None) is not None:
            # load action statistics for the denormalization
            self.action_mean, self.action_std = self._load_action_stats(
                config.checkpoint_path)
        else:
            self.action_mean, self.action_std = None, None

        self.image_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize([224, 224]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5), (0.5, 0.5))])

    def _load_action_stats(self, checkpoint_path):
        checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
        infos_path = os.path.join(checkpoint_dir, 'info.json')
        stats = json.load(open(infos_path, 'r'))['statistics'][0]
        # get network dataset statistics
        mean, std = {}, {}
        for action_key in self.action_keys:
            mean[action_key], std[action_key] = stats['action'][action_key]
        return mean, std

    @property
    def observation_space(self):
        if self.env_name == 'UR5-BowlEnv-v0':
            return Box(-np.inf, np.inf, (19,), dtype=np.float)
        elif self.env_name == 'UR5-SaladEnv-v0':
            num_cups, num_drops = self.env.unwrapped.scene._n_cups, self.env.unwrapped.scene._num_drops
            num_features = 32 + 10 * num_cups + 3 * num_drops * num_cups
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)
        elif 'Cam' in self.env_name:
            if self.observation_type == 'depth':
                observation_dim = self.num_frames * 1
            elif self.observation_type == 'rgbd':
                observation_dim = self.num_frames * 4
            else:
                raise NotImplementedError
            # return Box(-np.inf, np.inf, (observation_dim, 240, 320), dtype=np.float)
            return Box(-np.inf, np.inf, (observation_dim, 224, 224), dtype=np.float)

    @property
    def action_space(self):
        if not self.use_bcrl_setup:
            return Discrete(self.num_skills)
        else:
            return Box(-np.inf, np.inf, (self.dim_skill_action,), dtype=np.float)

    def _extract_obs(self, obs_dict):
        if self.observation_type == 'depth':
            obs = obs_dict['depth0'][None]
        elif self.observation_type == 'rgbd':
            depth = obs_dict['depth0'][..., None]
            rgb = obs_dict['rgb0']
            rgbd = np.concatenate((depth, rgb), axis=2)
            obs = np.rollaxis(rgbd, axis=2)
        else:
            raise NotImplementedError
        return obs


    def _obs_dict_to_numpy(self, observs):
        if 'Cam' not in self.env_name:
            if isinstance(observs, dict):
                # after reset
                obs = observs
            else:
                # after any step
                obs = observs[-1]
            observation = np.array([])
            obs_sorted = OrderedDict(sorted(obs.items(), key=lambda t: t[0]))
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
            if not self.use_bcrl_setup:
                if isinstance(observs, dict):
                    obs_big = self._extract_obs(observs)
                    obs_big = np.tile(obs_big, 3).reshape((240, 320, -1))
                    # obs_big: 240 x 320 x 3(12)
                    obs_big = np.moveaxis(obs_big, 2, 0)
                    # obs_big: 3(12) x 240 x 320
                else:
                    obs_big = np.concatenate([
                        self._extract_obs(obs) for obs in observs[-self.num_frames:]])
                    # obs_big: 3(12) x 240 x 320
            else:
                obs_big = self._extract_obs(observs)
            observation = []
            for obs_channel in obs_big:
                # 1) transforms.ToPILImage expects 3D tensor, so we use [None]
                # 2) transforms.ToPILImage expects image between 0. and 1.
                # 3) we remove the extra dim with [0] afterwards
                obs_transformed = self.image_transform(obs_channel[..., None])[0].numpy()
                observation.append(obs_transformed)
            observation = np.stack(observation)
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

    def _get_action_dict(self, action):
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
                if self.action_mean and self.action_std:
                    action_value *= self.action_std[action_key]
                    action_value += self.action_mean[action_key]
                else:
                    assert self.use_direct_actions
                action_dict[action_key] = action_value
            else:
                action_value = -1 + 2 * (action[0] > action[1])
                action_dict['grip_velocity'] = action_value * 2
        return action_dict

    def _print_action(self, action):
        if not self.use_bcrl_setup:
            print('master action: {}'.format(action))
        # in bcrl setup we do not print the environment actions

    def _filter_action(self, action):
        # the scripts of the mime env seems to always return both IK and joints velocities
        # we filter the action depending on the action_space of the mime environment
        action_filtered = {}
        for action_key, action_value in action.items():
            if action_key in self.env.action_space.spaces.keys():
                action_filtered[action_key] = action_value
        return action_filtered

    def _scripted_step(self, action):
        action_chain = self.env.unwrapped.scene.script_subtask(action)
        if not isinstance(action_chain, itertools.chain):
            action_chain = itertools.chain(*action_chain)
        action_dict = self._get_null_action_dict()
        action_dict_null = deepcopy(action_dict)
        observs, rewards = [], []
        # print('RL action = {}'.format(action))
        if self._skip_unused_obs:
            # turn off the rendering
            self.env.unwrapped._observe = False
        for timestep in range(self.timescale):
            if self._skip_unused_obs and timestep == self.timescale - self.num_frames:
                # turn on the rendering for the last 3 frames
                self.env.unwrapped._observe = True
            action_update = self._filter_action(next(action_chain, action_dict_null))
            action_dict.update(action_update)
            obs, rew_step, done, info = self.env.step(action_dict)
            observs.append(obs)
            rewards.append(rew_step)
            if done:
                if self._skip_unused_obs and timestep < self.timescale - 1:
                    # oh shit, we did not render all the observations!
                    obs = self.env.unwrapped._get_observation(self.env.unwrapped.scene)
                    for i in range(min(self.timescale - timestep - 1, self.num_frames)):
                        observs.append(obs)
                break
        observation = self._obs_dict_to_numpy(observs)
        if self._skip_unused_obs:
            # turn on the rendering again
            self.env.unwrapped._observe = True
        # reward max pooling (mostly for the failure happening after we pour the second cup)
        reward = max(rewards)
        return observation, reward, done, info

    def _bcrl_step(self, action):
        # we do not want to put the resnet inside the environment process
        # so the temporal abstraction of the master is in the main.py
        # here we just take care that the last 3 depths are returned
        action_dict = self._get_action_dict(action)
        obs, reward, done, info = self.env.step(action_dict)
        observation = self._obs_dict_to_numpy(obs)
        return observation, reward, done, info

    def step(self, action):
        if self._render:
            self._print_action(action)
        if not self.use_bcrl_setup:
            observation_tuple = self._scripted_step(action)
        else:
            observation_tuple = self._bcrl_step(action)
        if self._render:
            failure_message = observation_tuple[3]['failure_message']
            if len(failure_message) > 0:
                print('env[{}] failed: {}'.format(self._id, failure_message))
        return observation_tuple

    def reset(self):
        if self._render:
            print('env is reset')
        obs = self.env.reset()
        return self._obs_dict_to_numpy(obs)

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode):
        self.env.render(mode)

    def close(self):
        self.env.close()
