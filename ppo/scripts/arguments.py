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
                        help='whether to render the training and the evaluation')
    parser.add_argument('--pudb', action='store_true', default=False,
                        help='whether to stop the execution for manual commands')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--dask-batch-size', type=int, default=8,
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
    # BC stuff
    parser.add_argument('--augmentation', type=str, default='',
                        help='which data augmentation to use for the frames')
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
    parser.add_argument('--num-skills', type=int, default=4,
                        help='number of skills')
    parser.add_argument('--no-skip-unused-obs', action='store_true', default=False,
                        help='whether to render the observations not used by master (scripted setup)')
    # BC skills
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='if specified, load the networks weights from the file')
    # evaluation
    parser.add_argument('--num-eval-episodes', type=int, default=8,
                        help='number of episodes to use in evaluation')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--eval-max-length-factor', type=float, default=1.0,
                        help='horizon for eval episodes is the max length (train) multiplied by this')
    parser.add_argument('--eval-offline', action='store_true', default=False,
                        help='whether to only save the eval checkpoints for the offline evaluation')
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

    args = parser.parse_args()
    args.recurrent_policy = False  # turn off recurrent policies support
    args.skip_unused_obs = not args.no_skip_unused_obs  # only works for the scripted setup

    return args
