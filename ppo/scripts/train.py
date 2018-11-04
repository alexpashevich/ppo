import os
import time
import torch
import numpy as np
from collections import deque
from gym.spaces import Discrete


import ppo.algo as algo
import ppo.tools.utils as utils
import ppo.tools.envs as envs
import ppo.tools.log as log
import ppo.tools.stats as stats
from ppo.scripts.arguments import get_args
from ppo.tools.envs import make_vec_envs
from ppo.parts.model import Policy, MasterPolicy
from ppo.parts.storage import RolloutStorage


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
        envs.make_env(args.env_name, args.seed, 0, args.add_timestep, False, args)])
    env_render_eval = SubprocVecEnv([
        envs.make_env(args.env_name, args.seed + args.num_processes, 0, args.add_timestep, False, args)])
    return env_render_train, env_render_eval


def create_policy(args, envs, device, action_space):
    PolicyClass = MasterPolicy if args.use_bcrl_setup else Policy
    policy = PolicyClass(envs.observation_space.shape, action_space, base_kwargs=vars(args))
    if args.checkpoint_path:
        utils.load_from_checkpoint(policy, args.checkpoint_path, device)
    return policy


def create_agent(args, policy):
    agent = algo.PPO(
        policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    return agent


def _perform_actions(action_sequence, observation, policy, envs, env_render, args):
    for action in action_sequence:
        master_action_numpy = [[action] for _ in range(observation.shape[0])]
        master_action = torch.Tensor(master_action_numpy).int()
        if not args.use_bcrl_setup:
            observation, _, _, _ = envs.step(master_action)
            if env_render is not None:
                env_render.step(master_action[:1].numpy())
        else:
            observation, _, _, _ = utils.do_master_step(
                master_action, observation, args.timescale, policy, envs, env_render)


def main():
    args = get_args()
    render = args.render
    logdir = os.path.join(args.logdir, args.timestamp)
    # get the device before loading to enable the GPU/CPU transfer
    device = utils.get_device(args.device)
    print('Running the experiments on {}'.format(device))
    # try to load from a checkpoint
    loaded, loaded_tuple = utils.try_to_load_model(logdir)
    if loaded:
        policy, optimizer_state_dict, ob_rms, start_epoch, args = loaded_tuple
    utils.set_up_training(args)
    log.init_writers(os.path.join(logdir, 'train'), os.path.join(logdir, 'eval'))

    # create the parallel envs
    envs = create_envs(args, device)
    eval_envs = None
    # render_envs = create_render_env(args, device) if render else None
    env_render_train, env_render_eval = create_render_env(args, device) if render else (None, None)
    # create the policy
    # TODO: refactor this later
    action_space = Discrete(args.num_skills) if args.use_bcrl_setup else envs.action_space
    if not loaded:
        policy = create_policy(args, envs, device, action_space)
        start_epoch = 0
    policy.to(device)
    # create the PPO algo
    agent = create_agent(args, policy)

    if loaded:
        # load normalization and optimizer statistics
        utils.load_ob_rms(ob_rms, envs)
        utils.load_optimizer(agent.optimizer, optimizer_state_dict, device)

    num_updates = int(args.num_frames) // args.num_frames_per_update // args.num_processes
    num_master_steps_per_update = args.num_frames_per_update // args.timescale
    rollouts = RolloutStorage(
        num_master_steps_per_update, args.num_processes, envs.observation_space.shape,
        action_space, policy.recurrent_hidden_state_size)

    obs = envs.reset()
    if env_render_train:
        env_render_train.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    stats_global, stats_local = stats.init(args)
    start = time.time()

    perform_actions = lambda seq: _perform_actions(
        seq, obs, policy, envs, env_render_train, args)
    if args.pudb:
        # you can call, e.g. perform_actions([0, 0, 1, 2, 3]) in the terminal
        import pudb; pudb.set_trace()
    for epoch in range(start_epoch, num_updates):
        print('Starting epoch {}'.format(epoch))
        for step in range(num_master_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Observe reward and next obs
            # TODO: refactor
            if not args.use_bcrl_setup:
                obs, reward, done, infos = envs.step(action)
                if env_render_train is not None:
                    env_render_train.step(action[:1].numpy())
            else:
                obs, reward, done, infos = utils.do_master_step(
                    action, rollouts.obs[step], args.timescale, policy, envs, env_render_train)

            stats_global, stats_local = stats.update(
                stats_global, stats_local, reward, done, infos, args)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        # TODO: maybe add some kind of assert for resnet layers to stay unchanged after the update?
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if epoch % args.save_interval == 0:
            log.save_model(logdir, policy, agent.optimizer, epoch, device, envs, args)

        total_num_env_steps = (epoch + 1) * args.num_processes * args.num_frames_per_update

        if epoch % args.log_interval == 0 and len(stats_global['length']) > 1:
            log.log_train(
                total_num_env_steps, start, stats_global, action_loss,
                value_loss, dist_entropy)

        is_eval_time = (epoch > 0 and epoch % args.eval_interval == 0) or render
        if (args.eval_interval and len(stats_global['length']) > 1 and is_eval_time):
            eval_envs, stats_eval = utils.evaluate(
                policy, args, device, envs, eval_envs, env_render_eval)
            log.log_eval(total_num_env_steps, stats_eval)
            if epoch % (args.save_interval * args.eval_interval) == 0:
                log.save_model(logdir, policy, agent.optimizer, epoch, device, envs, args, eval=True)


if __name__ == "__main__":
    main()
