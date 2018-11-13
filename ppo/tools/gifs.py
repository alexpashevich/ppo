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


def update(gifs_global, gifs_local, master_action, done_now, done_before, envs_history):
    for env_id in range(master_action.shape[0]):
        if not done_before[env_id]:
            skill_tuple = envs_history['observations'][env_id], envs_history['skill_actions'][env_id]
            for obs_skill, action_skill in zip(*skill_tuple):
                frame = np.array((0.5 + obs_skill[-1:] * 0.5) * 255, dtype=np.uint8)
                gifs_local[env_id]['frames'].append(frame)
                gifs_local[env_id]['skill_actions'].append(action_skill)
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
    if gifs_list_of_dict is None:
        return
    print('Writing the gifs to {}'.format(gifs_dir))
    for idx_gif, gif_dict in enumerate(gifs_list_of_dict):
        if gif_dict is None:
            print('gif_dict is None')
            continue
        frames = gif_dict['frames']
        master_actions = gif_dict['master_actions']
        skill_actions = gif_dict['skill_actions']
        gif_name = '{:02}.gif'.format(idx_gif)
        if len(frames):
            try:
                videos.write_video(frames, os.path.join(gifs_dir, gif_name))
                np.savez(os.path.join(gifs_dir, '{:02}.npz'.format(idx_gif)),
                        master_actions=master_actions,
                        skill_actions=skill_actions)
            except:
                print('was not able to write the gif')
    print('Wrote {} gifs'.format(len(gifs_list_of_dict)))
