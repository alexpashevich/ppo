import os

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete
from copy import deepcopy
import itertools

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_


try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

from mime.agent.agent import Agent

class MiMEEnv(object):
    def __init__(self, env_name, config, id=0):
        self.env = gym.make(env_name)
        self.num_skills = config.num_skills
        self.timescale = config.timescale
        self._render = config.render and id == 0
        # some copypasting
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        # BowlEnv-v0 specific
        assert len(self.env.observation_space.spaces) == 7
        # activate rendering
        if self._render:
            scene = self.env.unwrapped.scene
            scene.renders(True)

    @property
    def observation_space(self):
        return Box(-np.inf, np.inf, (19,), dtype=np.float)

    @property
    def action_space(self):
        return Discrete(self.num_skills)

    def _obs_dict_to_numpy(self, obs):
        observation = (obs['tool_position'] + obs['linear_velocity'] + (obs['grip_velocity'],) +
                       obs['cube_position'] + obs['bowl_position'] +
                       tuple(obs['distance_to_cube']) + tuple(obs['distance_to_bowl']))
        return np.array(observation)

    def _print_action(self, action):
        if action == 0:
            print('master action: go to cube and release')
        elif action == 1:
            print('master action: go down')
        elif action == 2:
            print('master action: grasp')
        elif action == 3:
            print('master action: go up')
        elif action == 4:
            print('master action: go to bowl and release')

    def step(self, action):
        reward = 0
        if self._render:
            self._print_action(action)
        action_scripts = self.env.unwrapped.scene.script_subtask(action)
        action_chain = itertools.chain(*action_scripts)
        action_dict = {
            'linear_velocity': [0, 0, 0],
            'angular_velocity': [0, 0, 0],
            'grip_velocity': 0}
        action_dict_null = deepcopy(action_dict)
        for i in range(self.timescale):
            action_dict.update(next(action_chain, action_dict_null))
            obs, rew_step, done, info = self.env.step(action_dict)
            reward += rew_step
            end_of_episode = self.env._elapsed_steps >= self.env._max_episode_steps
            # i think that this is not necessary, doublecheck later
            done = done or end_of_episode
            if done:
                break
        observation = self._obs_dict_to_numpy(obs)
        reward /= self.timescale
        return observation, reward, done, info

    def reset(self):
        if self._render:
            print('env is reset')
        obs = self.env.reset()
        return self._obs_dict_to_numpy(obs)

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode):
        self.env.render(mode)

def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets, env_config):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif 'UR5' in env_id:
            print('creating MiME env with id {}'.format(rank))
            env = MiMEEnv(env_id, env_config, rank)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        # TODO: remove the first condition (added for MiME)
        if obs_shape and len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
                  device, allow_early_resets, num_frame_stack=None, env_config=None):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets, env_config)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # TODO: MiME is never normalized
    if envs.observation_space.shape and len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif envs.observation_space.shape and len(envs.observation_space.shape) == 3:
        # TODO: does not work for MiME
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
