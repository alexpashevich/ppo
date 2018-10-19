import torch

import copy
import os
import time
import numpy as np

from tensorboardX import SummaryWriter

from utils import get_vec_normalize

train_writer, eval_writer = None, None

def init_writers(train_logdir, eval_logdir):
    global train_writer, eval_writer
    train_writer = SummaryWriter(train_logdir)
    eval_writer = SummaryWriter(eval_logdir)

def add_summary(tag, value, iter, stage='train'):
    if stage == 'train':
        writer = train_writer
    elif stage == 'eval':
        writer = eval_writer
    else:
        raise NotImplementedError
    writer.add_scalar(tag, value, iter)


def log_train(total_steps, start, episode_rewards, action_loss, value_loss, entropy):
    end = time.time()
    print("Training after {} steps, FPS {}".format(
        total_steps, int(total_steps / (end - start))))
    print("Last {} training episodes: mean reward {:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
        len(episode_rewards), np.mean(episode_rewards),
        np.min(episode_rewards), np.max(episode_rewards)))
    add_summary('reward/mean_reward', np.mean(episode_rewards), total_steps)
    add_summary('reward/max_reward', np.max(episode_rewards), total_steps)
    add_summary('loss/action_loss', action_loss, total_steps)
    add_summary('loss/value_loss', value_loss, total_steps)
    add_summary('loss/entropy', entropy, total_steps)

def log_eval(episode_rewards, total_steps):
    print("Evaluation after {} steps using {} episodes: mean reward {:.5f}\n". format(
        total_steps, len(episode_rewards), np.mean(episode_rewards)))
    add_summary('reward/mean_reward', np.mean(episode_rewards), total_steps, 'eval')
    add_summary('reward/max_reward', np.max(episode_rewards), total_steps, 'eval')

def save_model(save_path, policy, epoch, cuda, envs, eval=False):
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    # A really ugly way to save a model to CPU
    save_model = policy
    if cuda:
        save_model = copy.deepcopy(policy).cpu()
    save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None), epoch]

    model_name = 'model_eval.pt' if eval else 'model.pt'
    torch.save(save_model, os.path.join(save_path, model_name))
