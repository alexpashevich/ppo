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

def main():
    device, logdir, eval_logdir = utils.set_up_training(args)
    log.init_writers(os.path.join(logdir, 'train'), os.path.join(logdir, 'eval'))

    render = args.render # save it for the evaluation
    args.render = False # we do not want to render the training envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, logdir, args.add_timestep, device, False,
                         env_config=args)
    policy, ob_rms, start_epoch = utils.try_to_load_policy(args, logdir, eval_logdir)
    if policy:
        utils.get_vec_normalize(envs).ob_rms = ob_rms
    else:
        start_epoch = 0
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
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    rewards_train.append(info['episode']['r'])

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
            log.save_model(logdir, policy, epoch, args.cuda, envs)

        total_num_env_steps = (epoch + 1) * args.num_processes * args.num_frames_per_update

        if epoch % args.log_interval == 0 and len(rewards_train) > 1:
            log.log_train(
                total_num_env_steps, start, rewards_train, action_loss, value_loss, dist_entropy)

        if (args.eval_interval is not None
                and len(rewards_train) > 1
                and epoch % args.eval_interval == 0):
            rewards_eval = utils.evaluate(policy, args, logdir, device, envs, render)
            log.log_eval(rewards_eval, total_num_env_steps)
            log.save_model(logdir, policy, epoch, args.cuda, envs, eval=True)


if __name__ == "__main__":
    main()
