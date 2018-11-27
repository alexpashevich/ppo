import gym
import numpy as np
import torch

from collections import deque

from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from ppo.parts.mime_env import MiMEEnv

def make_env(env_id, seed, rank, add_timestep, usued_allow_early_resets, env_config):
    def _thunk():
        if 'UR5' in env_id:
            print('creating MiME env with id {}'.format(rank))
            env = MiMEEnv(env_id, env_config, rank)
        else:
            env = gym.make(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, add_timestep,
                  device, unused_allow_early_resets, num_frame_stack=None, env_config=None):
    envs = [make_env(env_name, seed, i, add_timestep, unused_allow_early_resets, env_config)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if envs.observation_space.shape and len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, ret=False)
        # envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if 'Cam' in env_name:
        envs = VecPyTorchFrameStack(envs, 3, device)

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
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.stacked_obs = [deque(maxlen=nstack) for _ in range(venv.num_envs)]

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def _deque_to_tensor(self):
        obs_list = []
        for d in self.stacked_obs:
            obs_list.append(torch.cat(tuple(d)))
        tensor = torch.stack(obs_list)
        return tensor.to(self.device)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for (i, new) in enumerate(news):
            self.stacked_obs[i].append(obs[i])
            if new:
                for _ in range(self.nstack - 1):
                    self.stacked_obs[i].append(obs[i])
        return self._deque_to_tensor(), rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        for i in range(len(obs)):
            for _ in range(self.nstack):
                self.stacked_obs[i].append(obs[i])
        return self._deque_to_tensor()

    def close(self):
        self.venv.close()
