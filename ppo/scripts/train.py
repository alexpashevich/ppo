import os
import time
import torch
import imageio
import numpy as np
from tqdm import tqdm
from gym.spaces import Discrete, Box
from termcolor import colored
from argparse import Namespace

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
        args, bc_model, bc_statistics = load.bc_model(args, device)
    misc.seed_exp(args)
    log.init_writers(os.path.join(logdir, 'train'), eval_logdir=None)
    if args.write_gifs:
        args.gifdir = os.path.join(logdir, 'gifs')
        print('Gifs will be written to {}'.format(args.gifdir))
        if not os.path.exists(args.gifdir):
            os.mkdir(args.gifdir)
            for env_idx in range(args.num_processes):
                if not os.path.exists(os.path.join(args.gifdir, '{:02d}'.format(env_idx))):
                    os.mkdir(os.path.join(args.gifdir, '{:02d}'.format(env_idx)))

    # create the parallel envs
    envs = DaskEnv(args)

    # create the policy
    action_space = Discrete(len(args.skills_mapping))
    if loaded_dict:
        policy = loaded_dict['policy']
        policy.reset()
        start_step, start_epoch = loaded_dict['start_step'], loaded_dict['start_epoch']
    else:
        policy = utils.create_policy(args, envs, action_space, bc_model, bc_statistics)
        start_step, start_epoch = 0, 0
    policy.to(device)

    # create the PPO algo
    agent = utils.create_agent(args, policy)

    if loaded_dict:
        # load normalization and optimizer statistics
        envs.obs_running_stats = loaded_dict['obs_running_stats']
        load.optimizer(agent.optimizer, loaded_dict['optimizer_state_dict'], device)

    exp_dict = dict(
        device=device,
        envs=envs,
        start_epoch=start_epoch,
        start_step=start_step,
        action_space=action_space,
    )

    # create or load stats
    if loaded_dict:
        stats_global, stats_local = loaded_dict['stats_global'], loaded_dict['stats_local']
    else:
        stats_global, stats_local = stats.init(args.num_processes)

    return args, policy, agent, stats_global, stats_local, Namespace(**exp_dict)


def main():
    args = get_args()
    logdir = os.path.join(args.logdir, args.timestamp)
    args, policy, agent, stats_global, stats_local, exp_vars = init_training(args, logdir)
    misc.print_gpu_usage(exp_vars.device)
    envs = exp_vars.envs
    action_space_skills = Box(-np.inf, np.inf, (args.bc_args['dim_action'],), dtype=np.float)
    rollouts, obs = utils.create_rollout_storage(
        args, envs, policy, exp_vars.action_space, action_space_skills, exp_vars.device)
    start = time.time()

    if args.pudb:
        import pudb; pudb.set_trace()
    epoch, env_steps = exp_vars.start_epoch, exp_vars.start_step
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
                action, obs, reward, policy, envs, args)
            master_steps_done += np.sum(need_master_action)
            pbar.update(np.sum(need_master_action))

            stats_global, stats_local = stats.update(
                stats_global, stats_local, reward, done, infos, args)

            # If done then clean the history of observations.
            masks = {i: torch.FloatTensor([0.0] if done_ else [1.0]) for i, done_ in enumerate(done)}
            # check that the obs dictionary contains obs from all envs that will be stored in rollouts
            # we only store observations from envs which need a master action
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

        # saving the model
        if epoch % args.save_interval == 0:
            print('Saving the model after epoch {} for offline evaluation'.format(epoch))
            log.save_model(
                logdir, policy, agent.optimizer, epoch, env_steps, exp_vars.device, envs, args,
                stats_global, stats_local)

        # logging
        if epoch % args.log_interval == 0 and len(stats_global['length']) > 1:
            log.log_train(
                env_steps, start, stats_global, action_loss, value_loss, dist_entropy, epoch)
            misc.print_gpu_usage(exp_vars.device)
            if 'Cam' in args.env_name:
                imageio.imwrite(os.path.join(logdir, 'eval/obs_{}.png'.format(epoch)),
                                list(obs.values())[0][0].cpu().numpy())
        epoch += 1
        if env_steps > args.num_frames:
            print('Number of env steps reached the maximum number of frames')
            break


if __name__ == "__main__":
    main()
