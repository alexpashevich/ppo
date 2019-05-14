import os
import time
import torch
import numpy as np
from tqdm import tqdm
from gym.spaces import Discrete, Box
from termcolor import colored

import bc.utils.misc as bc_misc
from ppo.tools import misc
from ppo.tools import log
from ppo.tools import load
from ppo.tools import stats
from ppo.scripts import utils
from ppo.envs.dask import DaskEnv
from ppo.scripts.arguments import get_args


def init_training(args, logdir):
    # get the device before loading to enable the GPU/CPU transfer
    device = torch.device(bc_misc.get_device(args.device))
    print('Running the experiments on {}'.format(device))

    # try to load from a checkpoint
    loaded_dict = load.ppo_model(logdir, device)
    if loaded_dict:
        args = loaded_dict['args']
    else:
        args, bc_model, bc_statistics = load.bc_checkpoint(args, device)
    misc.seed_exp(args)
    log.init_writers(os.path.join(logdir, 'train'), os.path.join(logdir, 'eval'))

    # create the parallel envs
    envs_train, envs_eval = DaskEnv(args), None

    # create the policy
    action_space = Discrete(args.num_skills)
    if loaded_dict:
        policy = loaded_dict['policy']
        start_step, start_epoch = loaded_dict['start_step'], loaded_dict['start_epoch']
    else:
        policy = utils.create_policy(args, envs_train, action_space, bc_model, bc_statistics)
        start_step, start_epoch = 0, 0
    policy.to(device)

    # create the PPO algo
    agent = utils.create_agent(args, policy)

    if loaded_dict:
        # load normalization and optimizer statistics
        envs_train.obs_running_stats = loaded_dict['obs_running_stats']
        load.optimizer(agent.optimizer, loaded_dict['optimizer_state_dict'], device)

    all_envs = envs_train, envs_eval
    return args, device, all_envs, policy, start_epoch, start_step, agent, action_space


def main():
    args = get_args()
    logdir = os.path.join(args.logdir, args.timestamp)
    args, device, all_envs, policy, start_epoch, start_step, agent, action_space = init_training(
        args, logdir)
    misc.print_gpu_usage(device)
    envs_train, envs_eval = all_envs
    action_space_skills = Box(-np.inf, np.inf, (args.bc_args['dim_action'],), dtype=np.float)
    rollouts, obs = utils.create_rollout_storage(
        args, envs_train, policy, action_space, action_space_skills, device)

    stats_global, stats_local = stats.init(args.num_processes)
    start = time.time()

    if hasattr(policy.base, 'resnet'):
        assert_tensors = utils.create_frozen_skills_check(obs, policy)

    if args.pudb:
        # you can call, e.g. perform_actions([0, 0, 1, 2, 3]) in the terminal
        # utils.perform_actions([4,0,2,1,3,5,0,2,1,3], obs, policy, envs_train, args)
        utils.perform_actions([5,0,0,1,2,3,4,4,6,0,0,1,2,3], obs, policy, envs_train, args)
        import pudb; pudb.set_trace()
    epoch, env_steps = start_epoch, start_step
    reward = torch.zeros((args.num_processes, 1)).type_as(obs[0])
    need_master_action, policy_values_cache = np.ones((args.num_processes,)), None
    while True:
        print('Starting epoch {}'.format(epoch))
        master_steps_done = 0
        pbar = tqdm(total=args.num_master_steps_per_update * args.num_processes)
        while master_steps_done < args.num_master_steps_per_update * args.num_processes:
            value, action, action_log_prob, recurrent_hidden_states = utils.get_policy_values(
                policy,
                rollouts.get_last(rollouts.obs),
                rollouts.get_last(rollouts.actions),
                rollouts.get_last(rollouts.recurrent_hidden_states),
                rollouts.get_last(rollouts.masks),
                policy_values_cache,
                need_master_action)
            policy_values_cache = value, action, action_log_prob, recurrent_hidden_states

            # Observe reward and next obs
            obs, reward, done, infos, need_master_action = utils.do_master_step(
                action, rollouts.get_last(rollouts.obs), reward, policy, envs_train, args.hrlbc_setup)
            master_steps_done += np.sum(need_master_action)
            pbar.update(np.sum(need_master_action))

            stats_global, stats_local = stats.update(
                stats_global, stats_local, reward, done, infos, args)

            # If done then clean the history of observations.
            masks = {i: torch.FloatTensor([0.0] if done_ else [1.0]) for i, done_ in enumerate(done)}
            # check that the obs dictionary contains obs from all envs that will be stored in rollouts
            assert len(set(np.where(need_master_action)[0]).difference(obs.keys())) == 0
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                indices=np.where(need_master_action)[0])
            reward[np.where(done)] = 0
            env_steps += sum([info['length_after_new_action']
                              for info in np.array(infos)[np.where(need_master_action)[0]]])
        pbar.close()

        print('Environment steps per 1 master step (in average): {:.1f}'.format(
            env_steps / master_steps_done))
        print('Environment steps per process (in average): {:.1f} (max_length = {})'.format(
            env_steps / args.num_processes, args.max_length))
        if (env_steps / args.num_processes / args.max_length > 1.5 or
            env_steps / args.num_processes / args.max_length < 0.5):
            print(colored((
                'Ratio of average number of environment steps per max-length is {:.2f}, ' +
                'consider changing --max-length or --num-master-steps-per-update').format(
                    env_steps / args.num_processes / args.max_length), 'yellow'))

        # master policy training
        with torch.no_grad():
            next_value = policy.get_value_detached(
                rollouts.get_last(rollouts.obs),
                rollouts.get_last(rollouts.actions),
                rollouts.get_last(rollouts.recurrent_hidden_states),
                rollouts.get_last(rollouts.masks))
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if hasattr(policy.base, 'resnet'):
            utils.do_frozen_skills_check(policy, *assert_tensors)

        if epoch % args.save_interval == 0:
            log.save_model(
                logdir, policy, agent.optimizer, epoch, env_steps, device, envs_train, args)

        if epoch % args.log_interval == 0 and len(stats_global['length']) > 1:
            log.log_train(
                env_steps, start, stats_global, action_loss, value_loss, dist_entropy, epoch)
            misc.print_gpu_usage(device)

        is_eval_time = args.eval_interval > 0 and (epoch % args.eval_interval == 0)
        if len(stats_global['length']) > 0 and is_eval_time:
            log.save_model(
                logdir, policy, agent.optimizer, epoch, env_steps, device, envs_train, args)
            if not args.eval_offline:
                envs_eval, stats_eval = utils.evaluate(
                    policy, args, device, envs_train, envs_eval)
                log.log_eval(env_steps, stats_eval)
            else:
                print('Saving the model after epoch {} for the offline evaluation'.format(epoch))
        epoch += 1
        if env_steps > args.num_frames:
            print('Number of env steps reached the maximum number of frames')
            break


if __name__ == "__main__":
    main()
