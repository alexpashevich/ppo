import torch
import copy
import numpy as np

from ppo.algo.ppo import PPO
from ppo.parts.model import MasterPolicy
from ppo.parts.storage import RolloutStorage
from ppo.envs.dask import DaskEnv

from ppo.tools import stats
from ppo.tools import misc


def create_policy(args, envs, action_space, bc_model, bc_statistics):
    policy = MasterPolicy(
        envs.observation_space.shape,
        action_space,
        bc_model,
        bc_statistics,
        **vars(args))
    return policy


def create_agent(args, policy):
    agent = PPO(
        policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    return agent


def create_rollout_storage(
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


def create_frozen_skills_check(obs, policy):
    # GT to check whether the skills stay unchanged
    with torch.no_grad():
        test_tensor = misc.dict_to_tensor(obs)[0]
        test_master = np.random.randint(0, policy.base.num_skills, len(obs))
        policy.base.resnet.eval()
        unused_skills, features_check = policy.base.resnet(test_tensor)
        skills_check = policy.base(test_tensor, None, None, None, test_master)
    return test_tensor, test_master, features_check, skills_check


def do_frozen_skills_check(
        policy, test_tensor, test_master, feature_check, skills_check):
    # check if the skills are not changed by the the RL updates
    with torch.no_grad():
        unused_skills, features_after_upd = policy.base.resnet(test_tensor)
        skills_after_upd = policy.base(test_tensor, None, None, None, test_master)
    assert (features_after_upd == feature_check).all()
    assert (skills_after_upd == skills_check).all()


def evaluate(policy, args_train, device, envs_train, envs_eval):
    args = copy.deepcopy(args_train)
    args.render = False
    # make the evaluation horizon longer (if eval_max_length_factor > 1)
    args.max_length = int(args.max_length * args.eval_max_length_factor)
    args.dask_batch_size = int(args.num_eval_episodes / 2)
    num_processes = args.num_eval_episodes
    args.num_processes = num_processes
    args.seed += num_processes
    if envs_eval is None:
        envs_eval = DaskEnv(args)
        envs_eval.obs_running_stats = envs_train.obs_running_stats

    obs = envs_eval.reset()
    recurrent_hidden_states = {env_idx: torch.zeros(policy.recurrent_hidden_state_size, device=device)
                               for env_idx in range(num_processes)}
    masks = {env_idx: torch.zeros(1, device=device) for env_idx in range(num_processes)}
    stats_global, stats_local = stats.init(num_processes, eval=True)

    need_master_action, policy_values_cache = np.ones((num_processes,)), None
    reward = torch.zeros((num_processes, 1)).type_as(obs[0])
    if args.action_memory == 0:
        memory_actions = None
    else:
        memory_actions = {env_idx: -torch.ones((args.action_memory,)).type_as(obs[0])
                          for env_idx in range(num_processes)}
    print('Evaluating...')
    while len(stats_global['return']) < args.num_eval_episodes:
        with torch.no_grad():
            policy_values_cache = get_policy_values(
                policy,
                obs,
                {key: memory_actions[key] for key in obs.keys()},
                recurrent_hidden_states,
                masks,
                policy_values_cache,
                need_master_action,
                deterministic=True)
            action, recurrent_hidden_states = policy_values_cache[1], policy_values_cache[3]

        # Observe reward and next obs
        obs, reward, done, infos, need_master_action = do_master_step(
            action, obs, reward, policy, envs_eval, args)
        memory_actions = update_memory_actions(memory_actions, action, need_master_action, done)
        stats_global, stats_local = stats.update(
            stats_global, stats_local, reward, done, infos, args, overwrite_terminated=False)
        masks = {i: torch.FloatTensor([0.0] if done_ else [1.0]) for i, done_ in enumerate(done)}
    return envs_eval, stats_global


def get_policy_values(
        policy,
        obs,
        memory_actions,
        recurrent_states,
        masks,
        policy_values_cache,
        need_master_action,
        deterministic=False):
    ''' The function is sampling the policy actions '''
    with torch.no_grad():
        value_new, action_new, log_prob_new, recurrent_states_new = policy.act(
            obs, memory_actions, recurrent_states, masks, deterministic)

        if policy_values_cache is None:
            return value_new, action_new, log_prob_new, recurrent_states_new

        value, action, log_prob, recurrent_states = policy_values_cache
        for env_idx in np.where(need_master_action)[0]:
            value[env_idx] = value_new[env_idx]
            action[env_idx] = action_new[env_idx]
            log_prob[env_idx] = log_prob_new[env_idx]
            recurrent_states[env_idx] = recurrent_states_new[env_idx]
        return value, action, log_prob, recurrent_states


def maybe_merge_skills_3_and_4(
        action_master, need_master_action, done_envs, info_envs, skills_34_were_switched, args):
    if not args.merge_skills_3_and_4:
        return action_master, need_master_action, skills_34_were_switched
    # action_master_prev = {env_idx: torch.tensor(skill) for env_idx, skill in action_master.items()}
    for env_idx, skill_tensor in action_master.items():
        if skill_tensor.item() == 4:
            if skills_34_were_switched[env_idx] and need_master_action[env_idx]:
                # we manually switched 3 to 4 earlier
                # now we want to change the merge_switch_env_idxs to False
                # let the master choose another action
                skills_34_were_switched[env_idx] = 0
                # print('action 4 of env {} is done'.format(env_idx))
        elif skill_tensor.item() == 3:
            if need_master_action[env_idx]:
                assert skills_34_were_switched[env_idx] == 0
                if done_envs[env_idx]:
                    # env is done during the skill 3, do nothing
                    # let the master choose another action
                    # print('env {} is done during the skill 3'.format(env_idx))
                    continue
                if info_envs[env_idx]['silency_triggered']:
                    # env has triggered the silency condition for the skill 3
                    # let the master choose another action
                    # print('env {} skill 3 is silent'.format(env_idx))
                    continue
                # env has finished doing the skills 3, switch it to 4 and fool the need_master_action
                # print('switching for env {}'.format(env_idx))
                action_master[env_idx] = torch.tensor(skill_tensor + 1)
                need_master_action[env_idx] = 0
                skills_34_were_switched[env_idx] = 1
    # for env_idx, skill_tensor in action_master_prev.items():
    #     if skill_tensor.item() != action_master[env_idx].item():
    #         print('action_master = {} after skills merge'.format(
    #             [action.item() for env_idx, action in sorted(action_master.items())]))
    #         break

    return action_master, need_master_action, skills_34_were_switched


def update_action_master(action_master, skills_34_were_merged, args):
    if not args.merge_skills_3_and_4:
        return action_master
    action_master_copy = {}
    for env_idx, skill_tensor in action_master.items():
        if skill_tensor.item() < 3:
            action_master_copy[env_idx] = torch.tensor(skill_tensor)
        elif skill_tensor.item() > 3:
            action_master_copy[env_idx] = torch.tensor(skill_tensor + 1)
        else:
            if not skills_34_were_merged[env_idx]:
                # the master chose skill 3 which should not be changed
                action_master_copy[env_idx] = torch.tensor(skill_tensor)
            else:
                # we manually switched 3 to 4 and want to keep it for now
                action_master_copy[env_idx] = torch.tensor(skill_tensor + 1)
    return action_master_copy


def do_master_step(action_master, obs, reward_master, policy, envs, args, skills_34_were_switched):
    # we expect the action_master to have an action for each env
    # obs contains observations only for the envs that did a step and need a new skill action
    assert len(action_master.keys()) == envs.num_processes
    info_master = np.array([None] * envs.num_processes)
    done_master = np.array([False] * envs.num_processes)
    # print('action_master = {}'.format(
    #     [action.item() for env_idx, action in sorted(action_master.items())]))
    action_master = update_action_master(action_master, skills_34_were_switched, args)
    # print('action_master = {} after shift'.format(
    #     [action.item() for env_idx, action in sorted(action_master.items())]))
    while True:
        if args.hrlbc_setup:
            # get the skill action for env_idx in obs.keys()
            with torch.no_grad():
                action_skill_dict, env_idxs = policy.get_worker_action(action_master, obs)
        else:
            # create a dictionary out of master action values
            action_skill_dict = {}
            for env_idx, env_action_master in action_master.items():
                if env_idx in obs.keys():
                    action_skill_dict[env_idx] = {'skill': env_action_master}
        obs, reward_envs, done_envs, info_envs = envs.step(action_skill_dict)
        need_master_action = update_master_variables(
            num_envs=envs.num_processes,
            env_idxs=obs.keys(),
            envs_dict={'reward': reward_envs, 'done': done_envs, 'info': info_envs},
            master_dict={'reward': reward_master, 'done': done_master, 'info': info_master})
        action_master, need_master_action, skills_34_were_switched = maybe_merge_skills_3_and_4(
                action_master, need_master_action, done_envs, info_envs, skills_34_were_switched, args)
        if np.any(need_master_action):
            break

    # print('need_master_a = {}'.format(need_master_action))
    return (obs, reward_master, done_master, info_master, need_master_action, skills_34_were_switched)


def update_master_variables(num_envs, env_idxs, envs_dict, master_dict):
    '''' Returns a numpy array of 0/1 with indication which env needs a master action '''
    need_master_action = np.zeros((num_envs,))
    for env_idx in env_idxs:
        if envs_dict['info'][env_idx]['need_master_action']:
            need_master_action[env_idx] = 1
        if envs_dict['done'][env_idx]:
            need_master_action[env_idx] = 1
            master_dict['done'][env_idx] = True
        if need_master_action[env_idx]:
            master_dict['info'][env_idx] = envs_dict['info'][env_idx]
        master_dict['reward'][env_idx] += envs_dict['reward'][env_idx]
    return need_master_action


def update_memory_actions(memory_actions, action, need_master_action, done):
    ''' Updates the actions passed to the policy as a memory. '''
    if memory_actions is None:
        return memory_actions
    for env_idx in np.where(need_master_action)[0]:
        memory_actions[env_idx][:-1] = memory_actions[env_idx][1:]
        memory_actions[env_idx][-1] = action[env_idx][0]
    for env_idx, done_ in enumerate(done):
        if done_:
            memory_actions[env_idx][:] = -1.
    return memory_actions


def perform_skill_sequence(skill_sequence, observation, policy, envs, args):
    reward = torch.zeros((args.num_processes, 1)).type_as(observation[0])
    skill_counters = [0] * args.num_processes
    dones = [False] * args.num_processes
    while True:
        master_action_dict = {env_idx: torch.Tensor([skill_sequence[skill_counter]]).int()
                              for env_idx, skill_counter in enumerate(skill_counters)}
        observation, reward, done, _, need_master_action = do_master_step(
            master_action_dict, observation, reward, policy, envs, args)
        for env_idx, need_master_action_flag in enumerate(need_master_action):
            if need_master_action_flag:
                if skill_counters[env_idx] == len(skill_sequence) - 1:
                    skill_counters[env_idx] = 0
                    dones[env_idx] = True
                else:
                    skill_counters[env_idx] += 1
        print('rewards = {}'.format(reward[:, 0]))
        if all(dones):
            break
    envs.reset()
