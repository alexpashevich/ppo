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
import ppo.tools.stats as stats
import ppo.tools.gifs as gifs


def create_envs(args, device):
    # args.render = False
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
        args, envs_train, env_render_train, policy, action_space, action_space_skills, device):
    rollouts = RolloutStorage(
        args.num_master_steps_per_update, args.num_processes, envs_train.observation_space.shape,
        action_space, policy.recurrent_hidden_state_size)

    obs = envs_train.reset()
    if env_render_train:
        env_render_train.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    return rollouts, obs


def init_frozen_skills_check(obs, policy):
    # GT to check whether the skills stay unchanged
    with torch.no_grad():
        test_tensor = obs.clone()
        test_master = np.random.randint(0, policy.base.num_skills, len(obs))
        policy.base.resnet.eval()
        features_check = policy.base.resnet(test_tensor)
        skills_check = policy.base(test_tensor, None, None, test_master)
    return test_tensor, test_master, features_check, skills_check


def make_frozen_skills_check(policy, test_tensor, test_master, feature_check, skills_check):
    # check if the skills do not change by the RL training
    with torch.no_grad():
        features_after_upd = policy.base.resnet(test_tensor)
        skills_after_upd = policy.base(test_tensor, None, None, test_master)
    assert (features_after_upd == feature_check).all()
    assert (skills_after_upd == skills_check).all()


def evaluate(policy, args_train, device, train_envs_or_ob_rms, envs_eval, env_render=None):
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
    if env_render:
        env_render.reset()
    recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    stats_global, stats_local = stats.init(num_processes, eval=True)

    if args.save_gifs:
        gifs_global, gifs_local = gifs.init(num_processes)
    else:
        gifs_global = None

    need_master_action, prev_policy_outputs = np.ones((args.num_processes,)), None
    reward = 0
    print('Evaluating...')
    while len(stats_global['return']) < args.num_eval_episodes:
        with torch.no_grad():
            value_unused, action, action_log_prob_unused, recurrent_hidden_states = get_policy_values(
                policy, (obs, recurrent_hidden_states, masks),
                need_master_action, prev_policy_outputs, deterministic=True)
            prev_policy_outputs = value_unused, action, action_log_prob_unused, recurrent_hidden_states

        # Observe reward and next obs
        master_step_output = do_master_step_flex(
            action, obs, reward, policy, envs_eval,
            hrlbc_setup=args.hrlbc_setup,
            env_render=env_render,
            return_observations=args.save_gifs,
            evaluation=True)
        obs, reward, done, infos, need_master_action = master_step_output[:5]
        if args.save_gifs:
            # saving gifs only works for the BCRL setup
            gifs_global, gifs_local = gifs.update(
                gifs_global, gifs_local, action, done, stats_local['done_before'],
                master_step_output[-1])

        stats_global, stats_local = stats.update(
            stats_global, stats_local, reward, done, infos, need_master_action,
            args, overwrite_terminated=False)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
    return envs_eval, stats_global, gifs_global


def get_policy_values(
        policy, rollouts_or_explicit_tuple, need_master_action,
        prev_policy_outputs, deterministic=False):
    ''' The function is sampling the policy actions '''
    if isinstance(rollouts_or_explicit_tuple, (tuple, list)):
        # during evaluation we store the policy output in variables so we pass them directly
        obs, recurrent_states_input, masks = rollouts_or_explicit_tuple
    else:
        # during trainin we store the policy output in rollouts so we pass the rollouts
        rollouts = rollouts_or_explicit_tuple
        obs = rollouts.get_last(rollouts.obs)
        recurrent_states_input = rollouts.get_last(rollouts.recurrent_hidden_states)
        masks = rollouts.get_last(rollouts.masks)
    with torch.no_grad():
        if np.all(need_master_action):
            # each environment requires a master action
            value, action, log_prob, recurrent_states = policy.act(
                obs, recurrent_states_input, masks, deterministic)
        else:
            # only several environments require a master action
            # update only the values of those
            value, action, log_prob, recurrent_states = prev_policy_outputs
            indices = np.where(need_master_action)
            value[indices], action[indices], log_prob[indices], recurrent_states[indices] = policy.act(
                obs[indices], recurrent_states_input[indices], masks[indices], deterministic)
    return value, action, log_prob, recurrent_states


def do_master_step_flex(
        master_action, master_obs, master_reward, policy, envs,
        hrlbc_setup=False, env_render=None, return_observations=False, evaluation=False):
    print('master action = {}'.format(master_action[:, 0]))
    skill_obs = master_obs
    if return_observations:
        envs_history = {'observations': [[] for _ in range(master_action.shape[0])],
                        'skill_actions': [[] for _ in range(master_action.shape[0])]}
    master_infos = np.array([None] * master_action.shape[0])
    while True:
        if hrlbc_setup:
            # get the skill action
            with torch.no_grad():
                skill_action = policy.get_worker_action(master_action, skill_obs)
        else:
            # it is not really a skill action, but we use this name to simplify the code
            skill_action = master_action
        skill_obs, reward, done, infos = envs.step(skill_action)
        if env_render is not None:
            env_render.step(skill_action[:1].cpu().numpy())
        if return_observations:
            # TODO: rewrite in the numpy style
            for env_id, env_is_done in enumerate(done):
                # we do not want to record gifs after resets
                if not env_is_done:
                    envs_history['observations'][env_id].append(skill_obs[env_id].cpu().numpy())
                    envs_history['skill_actions'][env_id].append(skill_action[env_id].cpu().numpy())
                    # skill_actions_envs_list.append(skill_action.cpu().numpy())
        need_master_action = np.array([info['need_master_action'] for info in infos])
        need_master_action[np.where(done)] = True
        master_reward += reward
        master_infos[np.where(need_master_action)] = np.array(infos)[np.where(need_master_action)]
        if np.any(need_master_action):
            break
    return_tuple = [skill_obs, master_reward, done, master_infos, need_master_action]
    if return_observations:
        return_tuple += [envs_history]
    return return_tuple


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
            return_observations=True)
        # TODO: change the gifs writing. so far works only when envs are done after the actions
        if args.save_gifs:
            gifs_global, gifs_local = gifs.update(
                gifs_global, gifs_local, torch.tensor(master_action_numpy), done, done_before,
                observation_history)
            done_before = np.logical_or(done, done_before)
        print('reward = {}'.format(reward[:, 0]))
    if args.save_gifs:
        gifs.save(os.path.join(args.logdir, args.timestamp), gifs_global, epoch=-1)

