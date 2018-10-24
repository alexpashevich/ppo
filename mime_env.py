import gym
import itertools
import numpy as np

from collections import deque
from gym.spaces import Box, Discrete
from copy import deepcopy

from mime.agent.agent import Agent

SUPPORTED_MIME_ENVS = 'UR5-BowlEnv-v0', 'UR5-BowlCamEnv-v0'

class MiMEEnv(object):
    def __init__(self, env_name, config, id=0):
        assert env_name in SUPPORTED_MIME_ENVS

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.num_skills = vars(config).get('num_skills', 4)
        self.timescale = vars(config).get('timescale', 25)
        self._render = vars(config).get('render', False) and id == 0
        self.num_frames = 3  # use last 3 depth maps as an observation for BowlCamEnv
        self.last_observations = deque(maxlen=self.num_frames)
        # # TODO: implement the deque for non use_bcrl_setup
        # assert self.timescale >= self.num_frames  # the other case is not implemented yet
        # some copypasting
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        # activate rendering
        if self._render:
            scene = self.env.unwrapped.scene
            scene.renders(True)
        # bcrl related stuff
        self.use_bcrl_setup = vars(config).get('use_bcrl_setup', False)
        self.dim_skill_action = vars(config).get('dim_skill_action', 5)

    @property
    def observation_space(self):
        if self.env_name == 'UR5-BowlEnv-v0':
            return Box(-np.inf, np.inf, (19,), dtype=np.float)
        elif self.env_name == 'UR5-BowlCamEnv-v0':
            return Box(-np.inf, np.inf, (self.num_frames, 240, 320), dtype=np.float)

    @property
    def action_space(self):
        if not self.use_bcrl_setup:
            return Discrete(self.num_skills)
        else:
            return Box(-np.inf, np.inf, (self.dim_skill_action,), dtype=np.float)

    def _obs_dict_to_numpy(self, observs):
        if self.env_name == 'UR5-BowlEnv-v0':
            if isinstance(observs, dict):
                # after reset
                obs = observs
            else:
                # after any step
                obs = observs[-1]
            observation = (obs['tool_position'] + obs['linear_velocity'] + (obs['grip_velocity'],) +
                           obs['cube_position'] + obs['bowl_position'] +
                           tuple(obs['distance_to_cube']) + tuple(obs['distance_to_bowl']))
            observation = np.array(observation)
        elif self.env_name == 'UR5-BowlCamEnv-v0':
            if not self.use_bcrl_setup:
                if isinstance(observs, dict):
                    observation = np.tile(observs['depth0'], 3).reshape((240, 320, self.num_frames))
                    # observation: 240 x 320 x 3
                    observation = np.moveaxis(observation, 2, 0)
                    # observation: 3 x 240 x 320
                else:
                    observation = np.array([obs['depth0'] for obs in observs[-self.num_frames:]])
                    # observation: 3 x 240 x 320
            else:
                obs = observs['depth0']  # observs is always a dictionary with the last observation
                # append: up to 4 times if empty, we don't care and 1 time otherwise
                num_appends = 1 + self.num_frames - len(self.last_observations)
                for _ in range(num_appends):
                    self.last_observations.append(obs)
                observation = np.array(self.last_observations)
                # observation: 3 x 240 x 320
        return observation

    def _get_action_dict(self, action):
        # we need to multiply the actions by the max actions
        max_gripper_velocity = self.env.unwrapped.scene.max_gripper_velocity
        max_linear_velocity, max_angular_velocity = self.env.unwrapped.scene.max_tool_velocity

        action_dict = {
            'linear_velocity': [0, 0, 0],
            'angular_velocity': [0, 0, 0],
            'grip_velocity': 0}
        action_dict['grip_velocity'] = (-1 + 2 * (action[0] > action[1])) * max_gripper_velocity
        action_dict['linear_velocity'] = action[2:5] * max_linear_velocity
        if len(action) == 8:
            action_dict['angular_velocity'] = action[5:8] * max_angular_velocity
        return action_dict

    def _print_action(self, action):
        if self.use_bcrl_setup:
            # do not print the environment actions
            return
        if action == 0:
            print('master action: go to cube and release')
        elif action == 1:
            print('master action: go down and grasp')
        elif action == 2:
            print('master action: go up')
        elif action == 3:
            print('master action: go to bowl and release')

    def _scripted_step(self, action):
        action_scripts = self.env.unwrapped.scene.script_subtask(action)
        action_chain = itertools.chain(*action_scripts)
        action_dict = {
            'linear_velocity': [0, 0, 0],
            'angular_velocity': [0, 0, 0],
            'grip_velocity': 0}
        action_dict_null = deepcopy(action_dict)
        observs, reward = [], 0
        for _ in range(self.timescale):
            action_dict.update(next(action_chain, action_dict_null))
            obs, rew_step, done, info = self.env.step(action_dict)
            observs.append(obs)
            reward += rew_step
            if done:
                break
        observation = self._obs_dict_to_numpy(observs)
        reward /= self.timescale
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
            return self._scripted_step(action)
        else:
            return self._bcrl_step(action)

    def reset(self):
        if self._render:
            print('env is reset')
        self.last_observations.clear()
        obs = self.env.reset()
        return self._obs_dict_to_numpy(obs)

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode):
        self.env.render(mode)

    def close(self):
        self.env.close()
