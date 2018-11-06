import os
import numpy as np
import copy

from bc.utils import videos


def init(num_envs):
    gifs_global = [None] * num_envs
    gifs_local = []
    for _ in range(num_envs):
        gifs_local.append({'frames': [], 'skill_actions': [], 'master_actions': []})
    return gifs_global, gifs_local


def update(gifs_global, gifs_local, master_action, done_now, done_before, stack_obs, stack_act):
    for obs_skill, act_skill in zip(stack_obs, stack_act):
        for env_id, (obs_worker, act_worker) in enumerate(zip(obs_skill, act_skill)):
            if not done_before[env_id]:
                frame = np.array((0.5+obs_worker[-1:].cpu().numpy()*0.5)*255, dtype=np.uint8)
                gifs_local[env_id]['frames'].append(frame)
                gifs_local[env_id]['skill_actions'].append(act_worker)
    for env_id, master_action_env in enumerate(master_action.cpu().numpy()):
        if not done_before[env_id]:
            gifs_local[env_id]['master_actions'].append(master_action_env[0])

    idxs_done_now = np.where(done_now)[0]
    for env_id in idxs_done_now:
        if not done_before[env_id]:
            gifs_global[env_id] = copy.deepcopy(gifs_local[env_id])
            gifs_local[env_id] = {'frames': [], 'skill_actions': [], 'master_actions': []}
    return gifs_global, gifs_local


def save(logdir_path, gifs_list_of_dict, epoch):
    all_gifs_dir = os.path.join(logdir_path, 'gifs')
    if not os.path.exists(all_gifs_dir):
        os.mkdir(all_gifs_dir)
    gifs_dir = os.path.join(all_gifs_dir, 'epoch{}'.format(epoch))
    if not os.path.exists(gifs_dir):
        os.mkdir(gifs_dir)
    print('Writing the gifs to {}'.format(gifs_dir))
    for idx_gif, gif_dict in enumerate(gifs_list_of_dict):
        frames = gif_dict['frames']
        master_actions = gif_dict['master_actions']
        skill_actions = gif_dict['skill_actions']
        gif_name = '{:02}.gif'.format(idx_gif)
        videos.write_video(frames, os.path.join(gifs_dir, gif_name))
        np.savez(os.path.join(gifs_dir, '{:02}.npz'.format(idx_gif)),
                 master_actions=master_actions,
                 skill_actions=skill_actions)
    print('Wrote {} gifs'.format(len(gifs_list_of_dict)))
