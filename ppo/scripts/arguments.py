import argparse
import datetime
import json


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # general
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--env-name', default='UR5-BowlCamEnv-v0',
                        help='environment to train on (default: UR5-BowlCamEnv-v0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to render the training')
    parser.add_argument('--pudb', action='store_true', default=False,
                        help='whether to stop the execution for manual commands')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--dask-batch-size', type=int, default=None,
                        help='the envs will be stepped using the batch size (default: 8)')
    parser.add_argument('--num-frames', type=int, default=30e7,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--max-length', type=int, default=None,
                        help='episodes max length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which device to run the experiments on: cuda or cpu')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--input-type', type=str, default='depth',
                        help='type of input for the conv nets')
    parser.add_argument('--action-memory', type=int, default=0,
                        help='number of last actions to pass to the agent')
    # RL algorithm hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--ppo-epoch', type=int, default=5,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=16,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--num-master-steps-per-update', type=int, default=None,
                        help='number of forward steps in A2C (default: 5)')
    # loss and clippings
    parser.add_argument('--value-loss-coef', type=float, default=1.,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.05,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    # returns computation
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    # optimizer
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    # hieararchy
    parser.add_argument('--hrlbc-setup', action='store_true', default=False,
                        help='use the setup with pretrained with BC skills')
    parser.add_argument('--timescale', type=json.loads, default=None,
                        help='dict of timescales corresponding to each skill or the timescale value')
    # BC stuff
    parser.add_argument('--augmentation', type=str, default='',
                        help='which data augmentation to use for the frames')
    # BC skills
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='if specified, load the networks weights from the file')
    parser.add_argument('--check-skills-silency', action='store_true', default=False,
                        help='whether to check skill silency condition (only works for salad)')
    # full state stuff
    parser.add_argument('--mime-action-space', type=str, default=None,
                        help='number of last actions to pass to the agent')
    # logging
    parser.add_argument('--logdir', default='./logs/',
                        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument('--timestamp', type=str,
                        default=datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
                        help='timestep for a given training')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=2,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--write-gifs', action='store_true', default=False,
                        help='whether to write ALL environments gifs to $LOGDIR/gifs')
    # master head
    parser.add_argument('--master-type', type=str, default='conv',
                        help='set vision based master head type')
    parser.add_argument('--master-num-channels', type=int, default=64,
                        help='set master number of channels')
    parser.add_argument('--master-conv-filters', type=int, default=3,
                        help='set vision based master layers depth')
    # tiny little harmless flags
    parser.add_argument('--skills-mapping', type=json.loads, default=None,
                        help='e.g. {"1": [1, 2], "2": [3], "4": [4, 5, 6]}')

    args = parser.parse_args()
    assert args.skills_mapping is not None
    if args.skills_mapping is None:
        print('WARNING: skills_mapping is not specified')
    assert args.algo == 'ppo'
    if not isinstance(args.timescale, dict):
        print('WARNING: args.timescale is not a dict')
    args.recurrent_policy = False  # turn off recurrent policies support
    if args.dask_batch_size is None:
        args.dask_batch_size = max(1, int(args.num_processes / 2))
    if args.num_master_steps_per_update is None:
        print('WARNING: num_master_steps_per_update is not specified')

    return args
