import os
import time
from collections import deque
from gym.spaces import Discrete

import torch

import ppo.algo as algo
import ppo.tools.utils as utils
import ppo.tools.log as log
from ppo.scripts.arguments import get_args
from ppo.tools.envs import make_vec_envs
from ppo.parts.model import Policy, MasterPolicy
from ppo.parts.storage import RolloutStorage

def create_envs(args, device):
    args.render = args.render_train
    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma,
        None, args.add_timestep, device, False, env_config=args)
    return envs

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

def main():
    args = get_args()
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
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    rewards_train = deque([0]*100, maxlen=100)

    start = time.time()
    for epoch in range(start_epoch, num_updates):
        print('Starting epoch {}'.format(epoch))
        for step in range(num_master_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Observe reward and next obs
            if not args.use_bcrl_setup:
                obs, reward, done, infos = envs.step(action)
            else:
                # print('master action = {}'.format(action.cpu().numpy()))
                obs, reward, done, infos = utils.do_master_step(
                    action, rollouts.obs[step], args.timescale, policy, envs)

            for info in infos:
                if 'episode' in info.keys():
                    rewards_train.append(info['episode']['r'])

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

        if epoch % args.log_interval == 0 and len(rewards_train) > 1:
            log.log_train(
                total_num_env_steps, start, rewards_train, action_loss, value_loss, dist_entropy)

        if (args.eval_interval and len(rewards_train) > 1 and epoch % args.eval_interval == 0):
            rewards_eval = utils.evaluate(policy, args, logdir, device, envs, args.render_eval)
            # TODO: put the network in the eval mode
            # .eval and .train
            log.log_eval(rewards_eval, total_num_env_steps)
            if epoch % (args.save_interval * args.eval_interval) == 0:
                log.save_model(logdir, policy, agent.optimizer, epoch, device, envs, args, eval=True)


if __name__ == "__main__":
    main()
