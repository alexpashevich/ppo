from collections import deque
import numpy as np


ALL_STATS = 'return', 'length', 'fail'


def init(args, eval=False):
    if not eval:
        # in the beginning of the training we want to have 100 values as well
        return_deque = deque([0]*100, maxlen=100)
        success_deque = deque([0]*100, maxlen=100)
    else:
        return_deque = deque(maxlen=100)
        success_deque = deque(maxlen=100)
    stats_global = {'return': return_deque,
                    'length': deque(maxlen=100),
                    'fail': deque(maxlen=100),
                    'fail_joints': deque(maxlen=100),
                    'fail_workspace': deque(maxlen=100),
                    'success': success_deque}
    stats_local = {'return': np.array([0] * args.num_processes, dtype=np.float32),
                   'length': np.array([0] * args.num_processes, dtype=np.int32)}
    return stats_global, stats_local


def update(stats_g, stats_l, reward, done, infos, args):
    stats_l['return'] += reward[:, 0].numpy()
    stats_l['length'] += 1
    # append stats of the envs that are done (reset or fail)
    return_done = stats_l['return'][np.where(done)]
    stats_g['return'].extend(return_done)
    stats_g['length'].extend(stats_l['length'][np.where(done)] * args.timescale)
    success_done = (return_done > 0) + 0
    stats_g['success'].extend(success_done)
    infos_done = np.array(infos)[np.where(done)]
    fail_messages_done = [info['failure_message'] for info in infos_done]
    num_done = int(np.sum(done))
    num_fail = int(np.sum([len(m) > 0 for m in fail_messages_done]))
    num_fail_joints = int(np.sum(['Joint' in m for m in fail_messages_done]))
    num_fail_workspace = int(np.sum(['Workspace' in m for m in fail_messages_done]))
    stats_g['fail'].extend([1] * num_fail + [0] * (num_done - num_fail))
    stats_g['fail_joints'].extend([1] * num_fail_joints + [0] * (num_done - num_fail_joints))
    stats_g['fail_workspace'].extend([1] * num_fail_workspace + [0] * (num_done - num_fail_workspace))
    # zero out returns of the envs that are done (reset or fail)
    stats_l['length'][np.where(done)] = 0
    stats_l['return'][np.where(done)] = 0
    return stats_g, stats_l
