import os
import time
from collections import deque

import torch

# local imports
import algo, utils, log
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage

args = get_args()

num_updates = int(args.num_frames) // args.num_frames_per_update // args.num_processes
logdir, logfile, eval_logdir = utils.set_up_training(args)
# print_log = lambda log_str: utils.print_log(log_str, logfile)

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    log.init_writers(os.path.join(logdir, 'train'), os.path.join(logdir, 'eval'))

    # TODO: do not use it in the future
    # env_config = {'num_skills': args.num_skills,
    #               'timescale': args.timescale,
    #               'render': args.render}
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, logdir, args.add_timestep, device, False, env_config=args)

    policy = Policy(envs.observation_space.shape, envs.action_space,
                    base_kwargs={'recurrent': args.recurrent_policy})
    policy.to(device)

    agent = algo.PPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps, max_grad_norm=args.max_grad_norm)

    num_master_steps_per_update = args.num_frames_per_update // args.timescale
    rollouts = RolloutStorage(num_master_steps_per_update, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              policy.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque([0] * args.num_eval_episodes, maxlen=args.num_eval_episodes)

    start = time.time()
    for epoch in range(num_updates):
        for step in range(num_master_steps_per_update):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if epoch % args.save_interval == 0 and logdir != "":
            log.save_model(logdir, policy, args.cuda, envs)

        total_num_env_steps = (epoch + 1) * args.num_processes * args.num_frames_per_update

        if epoch % args.log_interval == 0 and len(episode_rewards) > 1:
            log.log_train(
                total_num_env_steps, start, episode_rewards, action_loss, value_loss, dist_entropy)

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and epoch % args.eval_interval == 0):
            episode_rewards = utils.evaluate(policy, args, logdir, device, envs)
            log.log_eval(episode_rewards, total_num_env_steps)


if __name__ == "__main__":
    main()
