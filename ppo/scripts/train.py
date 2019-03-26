import os
import time
import torch
import numpy as np
from gym.spaces import Discrete, Box

import bc.utils.misc as bc_misc
import ppo.tools.misc as misc
import ppo.tools.log as log
import ppo.tools.stats as stats
import ppo.tools.gifs as gifs
import ppo.scripts.utils as utils
from ppo.scripts.arguments import get_args


def init_training(args):
    render_cached = args.render
    logdir = os.path.join(args.logdir, args.timestamp)

    # get the device before loading to enable the GPU/CPU transfer
    device = bc_misc.get_device(args.device)
    print('Running the experiments on {}'.format(device))

    # try to load from a checkpoint
    loaded_tuple = misc.try_to_load_model(logdir)
    if loaded_tuple:
        policy, optimizer_state_dict, ob_rms, start_epoch, args = loaded_tuple
    misc.seed_torch(args)
    log.init_writers(os.path.join(logdir, 'train'), os.path.join(logdir, 'eval'))

    # create the parallel envs
    envs_train, envs_eval = utils.create_envs(args, device), None
    env_render_train, env_render_eval = None, None
    # if render_cached:
    #     env_render_train, env_render_eval = utils.create_render_env(args, device)
    args.render = render_cached

    # create the policy
    action_space = Discrete(args.num_skills)
    if not loaded_tuple:
        policy = utils.create_policy(args, envs_train, device, action_space, logdir)
        start_epoch = 0
    policy.to(device)

    # create the PPO algo
    agent = utils.create_agent(args, policy)

    if loaded_tuple:
        # load normalization and optimizer statistics
        misc.load_ob_rms(ob_rms, envs_train)
        misc.load_optimizer(agent.optimizer, optimizer_state_dict, device)

    all_envs = envs_train, envs_eval, env_render_train, env_render_eval
    return logdir, args, device, all_envs, policy, start_epoch, agent, action_space


def main():
    args = get_args()
    logdir, args, device, all_envs, policy, start_epoch, agent, action_space = init_training(args)
    envs_train, envs_eval, env_render_train, env_render_eval = all_envs
    action_space_skills = Box(-np.inf, np.inf, (args.dim_skill_action,), dtype=np.float)
    rollouts, obs, num_master_steps_per_update = utils.init_rollout_storage(
        args, envs_train, env_render_train, policy, action_space, action_space_skills, device)

    num_updates = int(args.num_frames) // args.num_frames_per_update // args.num_processes

    stats_global, stats_local = stats.init(args.num_processes)
    start = time.time()

    if hasattr(policy.base, 'resnet'):
        assert_tensors = utils.init_frozen_skills_check(obs, policy)

    if args.pudb:
        # you can call, e.g. perform_actions([0, 0, 1, 2, 3]) in the terminal
        # utils.perform_actions([4,0,2,1,3,5,0,2,1,3], obs, policy, envs_train, None, args)
        import pudb; pudb.set_trace()
    for epoch in range(start_epoch, num_updates):
        print('Starting epoch {}'.format(epoch))
        for step in range(num_master_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = utils.do_master_step(
                action, rollouts.obs[step], args.timescale, policy, envs_train,
                args.hrlbc_setup, env_render_train)
            obs = utils.reset_early_terminated_envs(envs_train, env_render_train, done, obs, device)

            stats_global, stats_local = stats.update(
                stats_global, stats_local, reward, done, infos, args)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        # master policy training
        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if hasattr(policy.base, 'resnet'):
            utils.make_frozen_skills_check(policy, *assert_tensors)

        if epoch % args.save_interval == 0:
            log.save_model(logdir, policy, agent.optimizer, epoch, device, envs_train, args, eval=True)

        total_num_env_steps = (epoch + 1) * args.num_processes * args.num_frames_per_update

        if epoch % args.log_interval == 0 and len(stats_global['length']) > 1:
            log.log_train(
                total_num_env_steps, start, stats_global, action_loss, value_loss, dist_entropy)

        is_eval_time = args.eval_interval > 0 and (epoch % args.eval_interval == 0)
        if args.render or (len(stats_global['length']) > 0 and is_eval_time):
            log.save_model(logdir, policy, agent.optimizer, epoch, device, envs_train, args, eval=True)
            if not args.eval_offline:
                envs_eval, stats_eval, gifs_eval = utils.evaluate(
                    policy, args, device, envs_train, envs_eval, env_render_eval)
                log.log_eval(total_num_env_steps, stats_eval)
                if gifs_eval:
                    gifs.save(logdir, gifs_eval, epoch)
            else:
                print('Saving the model after epoch {} for the offline evaluation'.format(epoch))


if __name__ == "__main__":
    main()
