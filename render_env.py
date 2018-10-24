import numpy as np
from argparse import Namespace

from mime_env import MiMEEnv


def main():
    env_config = {'render': True, 'timescale': 25, 'num_skills': 4}
    # env_config = {'render': False, 'timescale': 25, 'num_skills': 4}
    env = MiMEEnv('UR5-BowlEnv-v0', Namespace(**env_config))
    # env = MiMEEnv('UR5-BowlCamEnv-v0', Namespace(**env_config))
    env.reset()
    # import pudb; pudb.set_trace()
    # a = 5
    for i in range(100):
        env.step(0)
        env.step(0)
        env.step(1)
        env.step(2)
        _, _, done, info = env.step(3)
        if not done:
            print('WARNING: {}'.format(info))
        else:
            print('iteration {} is successful'.format(i))
        env.reset()

if __name__ == '__main__':
    main()
