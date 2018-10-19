from argparse import Namespace
from envs import MiMEEnv


def main():
    env_config = {'render': True, 'timescale': 25, 'num_skills': 4}
    env = MiMEEnv('UR5-BowlEnv-v0', Namespace(**env_config))
    env.reset()
    import pudb; pudb.set_trace()
    a = 5

if __name__ == '__main__':
    main()
