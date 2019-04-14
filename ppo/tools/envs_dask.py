import gym
import torch
import mime
import numpy as np

from dask.distributed import Client, LocalCluster, Pub, Sub
from gym.spaces import Box, Discrete
from collections import OrderedDict, deque

from bc.dataset import Frames, Actions
from ppo.tools import misc

SUPPORTED_MIME_ENVS = 'Bowl', 'Salad', 'SimplePour', 'SimplePourNoDrops'


class DaskEnv:
    def __init__(self, config, statistics=None):
        # TODO: implement statistics
        assert any([env_prefix in config.env_name for env_prefix in SUPPORTED_MIME_ENVS])

        self._read_config(config)
        self._init_dask()
        self.statistics = statistics

        print('Created DaskEnv with {} processes and batch size of {}'.format(
            self.num_processes, self.batch_size))

    def _read_config(self, config):
        # parsing the environment part of the config
        self.env_name = config.env_name
        self.num_processes = vars(config).get('num_processes', 8)
        self.batch_size = vars(config).get('dask_batch_size', int(self.num_processes / 2))
        assert self.batch_size <= self.num_processes
        self.device = config.device
        self.observation_type = vars(config).get('input_type', 'depth')
        if self.observation_type == 'depth':
            self.channels = ('depth',)
        elif self.observation_type == 'rgbd':
            self.channels = ('depth', 'rgb')
        self.action_keys = Actions.action_space_to_keys(
            vars(config).get('action_space', 'tool'),
            vars(config).get('dim_skill_action', 8) - 1)  # I use -1 because I count the griper as 2 vals
        self.max_length = vars(config).get('max_length', None)
        self.num_skills = vars(config).get('num_skills', 4)  # TODO: use it
        self.render = vars(config).get('render', False)
        self.hrlbc_setup = vars(config).get('hrlbc_setup', False)
        self.augmentation = vars(config).get('augmentation', '')
        self.num_frames_stacked = vars(config).get('num_channels', 1)
        self.compress_frames = vars(config).get('compress_frames', False)  # TODO: implement it
        # timescales for skills (hrlbc setup only)
        self.skills_timescales = vars(config).get('timescale', 50)  # TODO: implement
        if isinstance(self.skills_timescales, int):
            skills_timescales = {}
            for skill in range(self.num_skills):
                skills_timescales[str(skill)] = self.skills_timescales
            self.skills_timescales = skills_timescales

    def _init_dask(self):
        cluster = LocalCluster(n_workers=self.num_processes)
        client = Client(cluster)
        # always define publishers first then subscribers
        pubs_action = [Pub('env{}_action'.format(seed)) for seed in range(self.num_processes)]
        _ = client.map(self._env_routine,
                       [self.env_name]*self.num_processes,
                       range(self.num_processes))
        sub_obs = Sub('observations')
        self.pubs_action = pubs_action
        self.sub_obs = sub_obs

    def _env_routine(self, env_name, env_idx):
        def create_env(env_name, env_idx, max_length, render):
            env = gym.make(env_name)
            env.seed(env_idx)
            if hasattr(env.unwrapped.scene, 'set_rl_mode'):
                env.unwrapped.scene.set_rl_mode()
            else:
                print('WARNING: the scene does not have set_rl_mode function')
            if max_length is not None:
                env._max_episode_steps = max_length
            if render and env_idx == 0:
                env.unwrapped.scene.renders(True)
            return env

        def convert_obs(obs_dict, frames_stack):
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
                frames_stack.append(obs_im)
                while len(frames_stack) < frames_stack.maxlen:
                    frames_stack.append(obs_im)
                obs_tensor = Frames.dic_to_tensor(
                    frames_stack,
                    self.channels,
                    Frames.sum_channels(self.channels),
                    augmentation_str=self.augmentation)
            return obs_tensor

        def publish_obs(pub, obs_dict, frames_stack, env_idx):
            obs_tensor = convert_obs(obs_dict['observation'], frames_stack)
            obs_dict['observation'] = obs_tensor
            pub.put((obs_dict, env_idx))

        env = create_env(env_name, env_idx, self.max_length, self.render)
        pub_obs = Pub('observations')
        sub_action = Sub('env{}_action'.format(int(env_idx)))
        obs = env.reset()
        step_counter = 0
        frames_stack = deque(maxlen=self.num_frames_stacked)
        publish_obs(pub_obs, {'observation': obs}, frames_stack, env_idx)
        for action in sub_action:
            obs, reward, done, info = env.step(action)
            if len(info['failure_message']):
                print('env {} failure {}'.format(env_idx, info['failure_message']))
            step_counter = step_counter + 1 if not done else 0
            info['length'] = step_counter
            publish_obs(
                pub_obs,
                {'observation': obs, 'reward': reward, 'done': done, 'info': info},
                frames_stack,
                env_idx)
            if done:
                frames_stack.clear()

    def step(self, actions):
        for env_idx, env_action in actions.items():
            action_dict = Actions.tensor_to_dict(env_action, self.action_keys, self.statistics)
            for action_key, action_value in action_dict.items():
                if isinstance(action_value, torch.Tensor):
                    action_dict[action_key] = action_value.cpu().numpy()
            self.pubs_action[env_idx].put(action_dict)
        return self.get_obs_batch(self.batch_size)

    def get_obs_batch(self, batch_size):
        obs_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        count_envs = 0
        for env_dict, env_idx in self.sub_obs:
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            reward_dict[env_idx] = env_dict['reward']
            done_dict[env_idx] = env_dict['done']
            info_dict[env_idx] = env_dict['info']
            info_dict[env_idx]['need_master_action'] = True  # TODO: fix it
            info_dict[env_idx]['length_after_new_action'] = 1
            count_envs += 1
            if count_envs == batch_size:
                break
        return obs_dict, reward_dict, done_dict, info_dict

    def reset(self):
        # TODO: fix it: this is not really a reset function.
        # if it is executed after several steps, it will be buggy
        count_envs = 0
        obs_dict = {}
        for env_dict, env_idx in self.sub_obs:
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            count_envs += 1
            if count_envs == self.num_processes:
                break
        # TODO: do something like
        # def reset(self):
        #     self._need_master_action = True
        #     self._step_counter, self._step_counter_after_new_action = 0, 0
        #     self.frames_stack.clear()
        #     self._prev_script = None
        #     print('env {} is reset'.format(self._id))
        #     if self._render:
        #         print('env is reset')
        #     obs = self.env.reset()
        #     observation = self._process_obs(obs)
        #     return observation
        return obs_dict

    @property
    def observation_space(self):
        if 'Cam' in self.env_name:
            if self.observation_type == 'depth':
                observation_dim = 1 * self.num_frames_stacked
            elif self.observation_type == 'rgbd':
                observation_dim = 4 * self.num_frames_stacked
            else:
                raise NotImplementedError
            return Box(-np.inf, np.inf, (observation_dim, 224, 224), dtype=np.float)
        elif 'Bowl' in self.env_name:
            return Box(-np.inf, np.inf, (19,), dtype=np.float)
        elif 'Salad' in self.env_name:
            # num_cups = self.env.unwrapped.scene._num_cups
            # num_drops = self.env.unwrapped.scene._num_drops
            num_cups = 2
            num_drops = 2
            num_features = 32 + 10 * num_cups + 3 * num_drops * num_cups
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)
        elif 'SimplePour' in self.env_name:
            if 'NoDrops' in self.env_name:
                num_drops = 0
            else:
                num_drops = 5
            num_features = 16 + 3 * num_drops
            return Box(-np.inf, np.inf, (num_features,), dtype=np.float)
        else:
            raise NotImplementedError
