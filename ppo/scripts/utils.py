import torch

import copy
import os
import numpy as np

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.tools.envs import VecPyTorchFrameStack, make_vec_envs, make_env
from ppo.tools.misc import get_vec_normalize, load_from_checkpoint
from ppo.algo.ppo import PPO
from ppo.parts.model import MasterPolicy, HierarchicalPolicy
from ppo.parts.storage import RolloutStorage
import ppo.tools.stats as stats
import ppo.tools.gifs as gifs


def create_envs(args, device):
    args.render = False
    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma,
        args.add_timestep, device, False, env_config=args)
    return envs


def create_render_env(args, device):
    args.render = True
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    env_render_train = SubprocVecEnv([
        make_env(args.env_name, args.seed, 0, args.add_timestep, False, args)])
    args_eval = copy.deepcopy(args)
    # make the evaluation horizon longer (if eval_max_length_factor > 1)
    args_eval.max_length = int(args.max_length * args.eval_max_length_factor)
    env_render_eval = SubprocVecEnv([
        make_env(args.env_name, args.seed + args.num_processes, 0, args.add_timestep, False, args_eval)])
    return env_render_train, env_render_eval


def create_policy(args, envs, device, action_space):
    if not args.learn_skills:
        policy = MasterPolicy(envs.observation_space.shape, action_space, base_kwargs=vars(args))
    else:
        policy = HierarchicalPolicy(envs.observation_space.shape, action_space, base_kwargs=vars(args))
    if args.checkpoint_path:
        load_from_checkpoint(policy, args.checkpoint_path, device, args.learn_skills)
    return policy


def create_agent(args, policy):
    agent = PPO(
        policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    return agent


def init_rollout_storage(
        args, envs_train, env_render_train, policy, action_space, action_space_skills, device):
    num_master_steps_per_update = args.num_frames_per_update // args.timescale
    rollouts_master = RolloutStorage(
        num_master_steps_per_update, args.num_processes, envs_train.observation_space.shape,
        action_space, policy.recurrent_hidden_state_size)
    rollouts_skills = RolloutStorage(
        args.num_frames_per_update, args.num_processes, envs_train.observation_space.shape,
        action_space_skills, policy.recurrent_hidden_state_size)

    obs = envs_train.reset()
    if env_render_train:
        env_render_train.reset()
    for rollouts in (rollouts_master, rollouts_skills):
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)
    return rollouts_master, rollouts_skills, obs, num_master_steps_per_update


def init_frozen_skills_check(obs, policy):
    # GT to check whether the skills stay unchanged
    test_tensor = obs.clone()
    policy.base.resnet.eval()
    feat_check = policy.base.resnet(test_tensor)
    return test_tensor, feat_check


def make_frozen_skills_check(policy, test_tensor, feat_check):
    # check if the skills do not change by the RL training
    feat_after_upd = policy.base.resnet(test_tensor)
    # assert (skills_after_upd == skills_check).all() and (feat_after_upd == feat_check).all()
    assert (feat_after_upd == feat_check).all()


def evaluate(policy, args_train, device, train_envs_or_ob_rms, eval_envs, env_render=None):
    args = copy.deepcopy(args_train)
    args.render = False
    # make the evaluation horizon longer (if eval_max_length_factor > 1)
    args.max_length = int(args.max_length * args.eval_max_length_factor)
    num_processes = args.num_eval_episodes
    if eval_envs is None:
        eval_envs = make_vec_envs(
            args.env_name, args.seed + num_processes, num_processes,
            args.gamma, args.add_timestep, device, True, env_config=args)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        if 'RunningMeanStd' in str(type(train_envs_or_ob_rms)):
            ob_rms = train_envs_or_ob_rms
        else:
            ob_rms = get_vec_normalize(train_envs_or_ob_rms).ob_rms
        vec_norm.ob_rms = ob_rms

    obs = eval_envs.reset()
    if env_render:
        env_render.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    stats_global, stats_local = stats.init(num_processes, eval=True)

    if args.save_gifs:
        gifs_global, gifs_local = gifs.init(num_processes)
    else:
        gifs_global = None

    print('Evaluating...')
    while len(stats_global['return']) < args.num_eval_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obs
        master_step_output = do_master_step(
            action, obs, args.timescale, policy, eval_envs,
            hrlbc_setup=args.hrlbc_setup,
            env_render=env_render,
            return_observations=args.save_gifs,
            learn_skills=args.learn_skills,
            evaluation=True)
        obs, reward, done, infos = master_step_output[:4]
        if args.save_gifs:
            # saving gifs only works for the BCRL setup
            gifs_global, gifs_local = gifs.update(
                gifs_global, gifs_local, action, done, stats_local['done_before'],
                master_step_output[4])

        stats_global, stats_local = stats.update(
            stats_global, stats_local, reward, done, infos, args, overwrite_terminated=False)
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
    return eval_envs, stats_global, gifs_global


def do_master_step(
        master_action, master_obs, master_timescale, policy, envs, rollouts_skills=None,
        hrlbc_setup=False, env_render=None, return_observations=False, learn_skills=False,
        evaluation=False):
    print('master action = {}'.format(master_action[:, 0]))
    master_reward = 0
    skill_obs = master_obs
    if return_observations:
        envs_history = {'observations': [[] for _ in range(master_action.shape[0])],
                        'skill_actions': [[] for _ in range(master_action.shape[0])]}
    master_done = np.array([False] * master_action.shape[0])
    master_infos = np.array([None] * master_action.shape[0])
    for _ in range(master_timescale):
        if hrlbc_setup:
            # get the skill action
            # TODO: refactor
            with torch.no_grad():
                if not learn_skills:
                    skill_action = policy.get_worker_action(master_action, skill_obs)
                else:
                    skill_value, skill_action, skill_action_log_prob, _ = policy.act_skill(
                        skill_obs, master_action, None, None, deterministic=evaluation)
        else:
            # it is not really a skill action, but we use this name to simplify the code
            skill_action = master_action
        for env_id, done_before in enumerate(master_done):
            # might be not the beget the skill action
            # we apply the null action to it
            if done_before:
                skill_action[env_id] = 0
        skill_obs, reward, done, infos = envs.step(skill_action)
        # TODO: what if done???
        skill_masks = torch.FloatTensor([[0.0] if done_ or master_done_ else [1.0]
                                         for done_, master_done_ in zip(done, master_done)])
        if rollouts_skills is not None and learn_skills:
            rollouts_skills.insert(
                skill_obs,
                torch.zeros_like(skill_action_log_prob),
                skill_action,
                skill_action_log_prob,
                skill_value,
                reward,
                skill_masks,
                master_action)
        if env_render is not None:
            env_render.step(skill_action[:1].cpu().numpy())
        if return_observations:
            for env_id, (done_before, done_now) in enumerate(zip(master_done, done)):
                # we do not want to record gifs after resets
                if not done_before and not done_now:
                    envs_history['observations'][env_id].append(skill_obs[env_id].cpu().numpy())
                    envs_history['skill_actions'][env_id].append(skill_action[env_id].cpu().numpy())
                    # skill_actions_envs_list.append(skill_action.cpu().numpy())
        # we do not add the rewards after reset
        reward[np.where(master_done)] = 0
        master_reward += reward
        terminated_right_now = np.logical_and(np.logical_not(master_done), done)
        master_infos[np.where(terminated_right_now)] = np.array(infos)[np.where(terminated_right_now)]
        master_done = np.logical_or(master_done, done)
        if master_done.all():
            break
    if not return_observations:
        return skill_obs, master_reward, master_done, master_infos
    else:
        return skill_obs, master_reward, master_done, master_infos, envs_history


def reset_early_terminated_envs(envs, env_render, done, obs, device, num_frames=3):
    # TODO: replace this method with a flexible timescale
    # we use the master with a fixed timescale, so some envs receive the old master action after reset
    # we manually reset the envs that were terminated during the master step after it
    # TODO: it also resets the envs that were just reset
    done_idxs = np.where(done)[0]
    if isinstance(envs.venv.venv, SubprocVecEnv):
        # we have several envs
        remotes = envs.venv.venv.remotes
        for idx in done_idxs:
            remotes[idx].send(('reset', None))
    if env_render and done[0]:
        env_render.reset()
    if isinstance(envs.venv.venv, SubprocVecEnv):
        # we use several envs in a batch
        for idx in done_idxs:
            obs_numpy = remotes[idx].recv()
            obs_torch = torch.from_numpy(obs_numpy).float().to(device)
            if isinstance(envs, VecPyTorchFrameStack):
                # observations are images
                for _ in range(num_frames):
                    envs.stacked_obs[idx].append(obs_torch)
            else:
                # observations are full states, no need to stack last 3 states
                obs[idx] = obs_torch
        if isinstance(envs, VecPyTorchFrameStack):
            # observations are images
            obs = envs._deque_to_tensor()
    else:
        # DummyVecEnv is used, we have only one env
        if 0 in done_idxs:
            obs = envs.reset()
    return obs


def perform_actions(action_sequence, observation, policy, envs, env_render, args):
    # observation = envs.reset()
    # if env_render:
    #     env_render.reset()
    if args.save_gifs:
        gifs_global, gifs_local = gifs.init(args.num_processes)
        done_before = np.array([False] * args.num_processes, dtype=np.bool)
    else:
        gifs_global = None
    for action in action_sequence:
        master_action_numpy = [[action] for _ in range(observation.shape[0])]
        master_action = torch.Tensor(master_action_numpy).int()
        observation, reward, done, _, observation_history = do_master_step(
            master_action, observation, args.timescale, policy, envs,
            args.hrlbc_setup,
            env_render=env_render,
            return_observations=True,
            learn_skills=args.learn_skills)
        # TODO: change the gifs writing. so far works only when envs are done after the actions
        if args.save_gifs:
            gifs_global, gifs_local = gifs.update(
                gifs_global, gifs_local, torch.tensor(master_action_numpy), done, done_before,
                observation_history)
            done_before = np.logical_or(done, done_before)
        print('reward = {}'.format(reward[:, 0]))
    if args.save_gifs:
        gifs.save(os.path.join(args.logdir, args.timestamp), gifs_global, epoch=-1)

