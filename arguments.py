import argparse
import datetime


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # general
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--env-name', default='UR5-BowlEnv-v0',
                        help='environment to train on (default: UR5-BowlEnv-v0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to render the evaluation')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-frames', type=int, default=30e7,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    # RL algorithm hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--ppo-epoch', type=int, default=5,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=16,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--num-frames-per-update', type=int, default=300,
                        help='number of forward steps in A2C (default: 5)')
    # loss and clippings
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
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
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    # hieararchy
    parser.add_argument('--num-skills', type=int, default=4,
                        help='number of skills')
    parser.add_argument('--timescale', type=int, default=25,
                        help='master timescale')
    # evaluation
    parser.add_argument('--num-eval-episodes', type=int, default=32,
                        help='numer of episodes to use in evluation')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='eval interval, one eval per n updates (default: None)')
    # logging
    parser.add_argument('--logdir', default='./logs/',
                        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument('--timestamp', type=str,
                        default=datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
                        help='timestep for a given training')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='save interval, one save per n updates (default: 100)')

    args = parser.parse_args()
    args.recurrent_policy = False  # turn off recurrent policies support

    return args
