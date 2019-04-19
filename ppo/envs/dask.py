import torch
import numpy as np

from dask.distributed import Client, LocalCluster, Pub, Sub
from baselines.common.running_mean_std import RunningMeanStd
from gym.spaces import Box

import bc.utils.misc as bc_misc
from ppo.envs.mime import MimeEnv

SUPPORTED_MIME_ENVS = 'Bowl', 'Salad', 'SimplePour', 'SimplePourNoDrops'


class DaskEnv:
    def __init__(self, config):
        assert any([env_prefix in config.env_name for env_prefix in SUPPORTED_MIME_ENVS])
        self._read_config(config)
        self._init_dask(config)
        self.action_sent_flags = np.zeros(self.num_processes)
        if 'Cam' not in self.env_name:
            self.obs_running_stats = RunningMeanStd(shape=self.observation_space.shape)
        else:
            self.obs_running_stats = None
        print('Created DaskEnv with {} processes and batch size of {}'.format(
            self.num_processes, self.batch_size))

    def _read_config(self, config):
        # parsing the environment part of the config
        self.env_name = config.env_name
        self.num_processes = config.num_processes
        self.batch_size = config.dask_batch_size
        assert self.batch_size <= self.num_processes
        self.device = bc_misc.get_device(config.device)
        self.observation_type = config.input_type
        self.num_frames_stacked = 3

    def _init_dask(self, config):
        cluster = LocalCluster(n_workers=self.num_processes)
        client = Client(cluster)
        # always define publishers first then subscribers
        pub_out = [Pub('env{}_input'.format(env_idx)) for env_idx in range(self.num_processes)]
        env_args = []
        for env_idx in range(self.num_processes):
            env_config = dict(env_idx=env_idx)
            env_config.update(vars(config))
            env_args.append(env_config)
        _ = client.map(MimeEnv, env_args)
        sub_in = Sub('observations')
        self.pub_out = pub_out
        self.sub_in = sub_in

        # clean the dask pipes
        counter = 0
        for none in self.sub_in:
            counter += 1
            if counter == self.num_processes:
                break

    def step(self, actions):
        for env_idx, action_dict in actions.items():
            if self.action_sent_flags[env_idx] == 1:
                continue
            self.action_sent_flags[env_idx] = 1
            for action_key, action_value in action_dict.items():
                if isinstance(action_value, torch.Tensor):
                    action_dict[action_key] = action_value.cpu().numpy()
            self.pub_out[env_idx].put({'function': 'step',
                                       'action': action_dict})
        return self._get_obs_batch(self.batch_size)

    def _get_obs_batch(self, batch_size):
        obs_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        count_envs = 0
        for env_dict, env_idx in self.sub_in:
            assert self.action_sent_flags[env_idx] == 1
            self.action_sent_flags[env_idx] = 0
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            reward_dict[env_idx] = env_dict['reward']
            done_dict[env_idx] = env_dict['done']
            info_dict[env_idx] = env_dict['info']
            count_envs += 1
            if count_envs == batch_size:
                break
        return self._normalize_obs(obs_dict), reward_dict, done_dict, info_dict

    def reset(self):
        count_envs = 0
        for env_idx in range(self.num_processes):
            assert self.action_sent_flags[env_idx] == 0
            self.pub_out[env_idx].put({'function': 'reset'})
        obs_dict = {}
        for env_dict, env_idx in self.sub_in:
            obs_dict[env_idx] = env_dict['observation'].to(torch.device(self.device))
            count_envs += 1
            if count_envs == self.num_processes:
                break
        return self._normalize_obs(obs_dict)

    def _normalize_obs(self, obs_dict):
        return obs_dict
        if self.obs_running_stats:
            obs_numpy_list = []
            for env_idx, obs in sorted(obs_dict.items()):
                obs_numpy_list.append(obs.cpu().numpy())
            self.obs_running_stats.update(np.stack(obs_numpy_list))
            clipob = 10.
            epsilon = 1e-8
            obs_mean = torch.tensor(self.obs_running_stats.mean).type_as(obs)
            obs_var = torch.tensor(self.obs_running_stats.var).type_as(obs)
            for env_idx, obs in sorted(obs_dict.items()):
                obs = torch.clamp((obs - obs_mean) / torch.sqrt(obs_var + epsilon), -clipob, clipob)
                obs_dict[env_idx] = obs
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
