import torch
import copy
import os
import numpy as np
import pickle as pkl

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from shutil import copyfile
from io import BytesIO

from ppo.tools.envs import VecPyTorchFrameStack, make_vec_envs, make_env
from ppo.tools.misc import get_vec_normalize, load_from_checkpoint
from ppo.algo.ppo import PPO
from ppo.parts.model import MasterPolicy
from ppo.parts.storage import RolloutStorage
from ppo.tools.envs_dask import DaskEnv

import ppo.tools.stats as stats
import ppo.tools.misc as misc


def create_envs(args, device):
    # args.render = False
    # envs = make_vec_envs(
    #     args.env_name, args.seed, args.num_processes, args.gamma,
    #     args.add_timestep, device, False, env_config=args)
    envs = DaskEnv(args)
    return envs


def create_policy(args, envs, device, action_space, logdir):
    policy = MasterPolicy(envs.observation_space.shape, action_space, base_kwargs=vars(args))
    if args.checkpoint_path:
        load_from_checkpoint(policy, args.checkpoint_path, device)
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if os.path.exists(os.path.join(checkpoint_dir, 'info.json')):
            copyfile(os.path.join(checkpoint_dir, 'info.json'), os.path.join(logdir, 'info.json'))
    return policy


def create_agent(args, policy):
    agent = PPO(
        policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    return agent


def init_rollout_storage(
        args, envs_train, policy, action_space, action_space_skills, device):
    rollouts = RolloutStorage(
        args.num_master_steps_per_update,
        args.num_processes,
        envs_train.observation_space.shape,
        action_space,
        policy.recurrent_hidden_state_size,
        action_memory=args.action_memory)

    obs = envs_train.reset()
    rollouts.obs[0].copy_(misc.dict_to_tensor(obs)[0])
    rollouts.to(device)
    return rollouts, obs


def init_frozen_skills_check(obs, policy):
    # GT to check whether the skills stay unchanged
    with torch.no_grad():
        test_tensor = obs.clone()
        test_master = np.random.randint(0, policy.base.num_skills, len(obs))
        policy.base.resnet.eval()
        features_check = policy.base.resnet(test_tensor)
        skills_check = policy.base(test_tensor, None, None, None, test_master)
    return test_tensor, test_master, features_check, skills_check


def make_frozen_skills_check(
        policy, test_tensor, test_master, feature_check, skills_check):
    # check if the skills are not changed by the the RL updates
    with torch.no_grad():
        features_after_upd = policy.base.resnet(test_tensor)
        skills_after_upd = policy.base(test_tensor, None, None, None, test_master)
    assert (features_after_upd == feature_check).all()
    assert (skills_after_upd == skills_check).all()


def evaluate(policy, args_train, device, train_envs_or_ob_rms, envs_eval):
    args = copy.deepcopy(args_train)
    args.render = False
    # make the evaluation horizon longer (if eval_max_length_factor > 1)
    args.max_length = int(args.max_length * args.eval_max_length_factor)
    num_processes = args.num_eval_episodes
    if envs_eval is None:
        envs_eval = make_vec_envs(
            args.env_name, args.seed + num_processes, num_processes,
            args.gamma, args.add_timestep, device, True, env_config=args)

    vec_norm = get_vec_normalize(envs_eval)
    if vec_norm is not None:
        vec_norm.eval()
        if 'RunningMeanStd' in str(type(train_envs_or_ob_rms)):
            ob_rms = train_envs_or_ob_rms
        else:
            ob_rms = get_vec_normalize(train_envs_or_ob_rms).ob_rms
        vec_norm.ob_rms = ob_rms

    obs = envs_eval.reset()
    recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    stats_global, stats_local = stats.init(num_processes, eval=True)


    need_master_action, prev_policy_outputs = np.ones((num_processes,)), None
    reward = 0
    if args.action_memory == 0:
        last_actions = None
    else:
        last_actions = -torch.ones(num_processes, args.action_memory).type_as(obs)
    print('Evaluating...')
    while len(stats_global['return']) < args.num_eval_episodes:
        with torch.no_grad():
            if last_actions is None:
                last_actions_local = None
            else:
                # TODO: check the [0] here
                last_actions_local = last_actions[np.where(need_master_action)[0]]
            value_unused, action, action_log_prob_unused, recurrent_hidden_states = get_policy_values(
                policy,
                (obs, last_actions_local, recurrent_hidden_states, masks),
                need_master_action, prev_policy_outputs, deterministic=True)
            prev_policy_outputs = value_unused, action, action_log_prob_unused, recurrent_hidden_states

        # Observe reward and next obs
        master_step_output = do_master_step(
            action, obs, reward, policy, envs_eval,
            hrlbc_setup=args.hrlbc_setup,
            evaluation=True)
        obs, reward, done, infos, need_master_action = master_step_output[:5]
        if last_actions is not None:
            for env_idx in np.where(need_master_action)[0]:
                last_actions[env_idx, :-1] = last_actions[env_idx, 1:]
                last_actions[env_idx, -1] = action[env_idx, 0]
            for env_idx, done_ in enumerate(done):
                if done_:
                    last_actions[env_idx] = -1.

        stats_global, stats_local = stats.update(
            stats_global, stats_local, reward, done, infos, need_master_action,
            args, overwrite_terminated=False)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
    return envs_eval, stats_global


def get_policy_values(
        policy, rollouts_or_explicit_tuple, need_master_action,
        prev_policy_outputs, deterministic=False):
    ''' The function is sampling the policy actions '''
    if isinstance(rollouts_or_explicit_tuple, (tuple, list)):
        # during evaluation we store the policy output in variables so we pass them directly
        obs, last_actions, recurrent_states_input, masks = rollouts_or_explicit_tuple
    else:
        # during trainin we store the policy output in rollouts so we pass the rollouts
        rollouts = rollouts_or_explicit_tuple
        obs = rollouts.get_last(rollouts.obs)
        last_actions = rollouts.get_last(rollouts.actions, processes=np.where(need_master_action)[0])
        recurrent_states_input = rollouts.get_last(rollouts.recurrent_hidden_states)
        masks = rollouts.get_last(rollouts.masks)
    with torch.no_grad():
        if np.all(need_master_action):
            # each environment requires a master action
            value, action, log_prob, recurrent_states = policy.act(
                obs, last_actions, recurrent_states_input, masks, deterministic)
        else:
            # only several environments require a master action
            # update only the values of those
            value, action, log_prob, recurrent_states = prev_policy_outputs
            # TODO: check the [0] here
            indices = np.where(need_master_action)[0]
            value[indices], action[indices], log_prob[indices], recurrent_states[indices] = policy.act(
                obs[indices], last_actions, recurrent_states_input[indices],
                masks[indices], deterministic)
    return value, action, log_prob, recurrent_states


def do_master_step(
        master_action, master_obs_tensor, master_reward, policy, envs,
        hrlbc_setup=False, evaluation=False):
    # print('master action = {}'.format(master_action[:, 0]))
    skill_obs_dict, env_idxs = misc.tensor_to_dict(master_obs_tensor)
    num_envs = master_action.shape[0]
    master_infos, master_dones = np.array([None] * num_envs), np.array([False] * num_envs)
    while True:
        if hrlbc_setup:
            # get the skill action
            with torch.no_grad():
                skill_action_dict, env_idxs = policy.get_worker_action(master_action, skill_obs_dict)
                # TODO: put it back
                # skill_action = torch.cat([skill_action, master_action.float()], dim=1)
        else:
            # it is not really a skill action, but we use this name to simplify the code
            skill_action_dict, env_idxs = misc.tensor_to_dict(master_action, env_idxs)
        skill_obs_dict, reward_dict, done_dict, infos_dict = envs.step(skill_action_dict)
        # TODO: move to a separate function
        need_master_action = np.zeros((num_envs,))
        for env_idx in skill_obs_dict.keys():
            if infos_dict[env_idx]['need_master_action']:
                need_master_action[env_idx] = 1
            if done_dict[env_idx]:
                need_master_action[env_idx] = 1
                master_dones[env_idx] = True
            if need_master_action[env_idx]:
                master_infos[env_idx] = infos_dict[env_idx]
            master_reward[env_idx] += reward_dict[env_idx]
        if np.any(need_master_action):
            break
    return (skill_obs_dict, master_reward, master_dones, master_infos, need_master_action)


def perform_actions(action_sequence, observation, policy, envs, args):
    # observation = envs.reset()
    reward = 0
    for action in action_sequence:
        master_action_numpy = [[action] for _ in range(observation.shape[0])]
        master_action = torch.Tensor(master_action_numpy).int()
        observation, reward, done, _, need_master_action = do_master_step(
            master_action, observation, reward, policy, envs,
            args.hrlbc_setup)
        print('reward = {}'.format(reward[:, 0]))

