import os
import subprocess
import argparse
import glob
import shutil
from termcolor import colored

HOME = os.environ['HOME']
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='{}/Logs/agents'.format(HOME),
                    help='directory where to save the logs of the tests')
parser.add_argument('--resnet18-branch-checkpoint-path', type=str,
                    default='{}/Dumps/salad_noreturn/resnet18_featbranch_166.pth'.format(
                        HOME),
                    help='directory with a BC skills resnet18_featbranch checkpoint')
# parser.add_argument('--resnet18-checkpoint-path', type=str,
#                     default='{}/Dumps/bowlenv/randskills200/resnet18_current.pth'.format(HOME),
#                     help='directory with a BC skills resnet18 checkpoint')
args = parser.parse_args()

UNITTESTS = [
    ['full state training on CPU',
     'python -m ppo.scripts.train --env-name=UR5-BowlEnv-v0 --num-frames-per-update=50 --num-processes=2 --num-mini-batch=2 --timescale=25 --num-frames=100 --eval-interval=1 --timestamp=4H20 --logdir={}/unittest/test1/ --device=cpu'.format(args.logdir)],
    ['HRLBC training',
     'python -m ppo.scripts.train --env-name=UR5-SaladCamEnv-v0 --num-frames-per-update=10 --num-processes=2 --num-mini-batch=2 --timescale=5 --num-frames=20 --eval-interval=1 --hrlbc-setup --checkpoint-path={} --dim-skill-action=8 --num-skill-action-pred=4 --num-skills=6 --archi=resnet18_featbranch --input-type=rgbd --timestamp=4H20 --logdir={}/unittest/test2/'.format(args.resnet18_branch_checkpoint_path, args.logdir)],
    ['loading the policy with enjoy.py',
     'python -m ppo.scripts.enjoy --load-path={}/unittest/test2/4H20 --num-whiles=1 --no-render'.format(args.logdir)]]
    # ['test skills of a BC checkpoint',
    #  'python -m ppo.scripts.test_skills --env-name=UR5-SaladCamEnv-v0 --num-processes=2 --num-episodes=2 --checkpoint-path={} --dim-skill-action=8 --num-skill-action-pred=4 --num-skills=6 --input-type=rgbd --timescale=4 --archi=resnet18_featbranch --action-sequence=[0,0,1]'.format(args.resnet18_branch_checkpoint_path)]]


def run_test(name, command):
    try:
        print(colored('Running TEST {}'.format(name), 'yellow'))
        print(colored(command, 'yellow'))
        subprocess.check_call(command.split(' '))
        print(colored('TEST "{}" OK\n'.format(name), 'green'))
        return 1
    except Exception as e:
        print(colored('TEST "{}" FAILED with Exception {}\n'.format(name, e), 'green'))
        return 0


def main():
    # remove the test directories first
    for test_dir in glob.glob('{}/unittest/test*'.format(args.logdir)):
        shutil.rmtree(test_dir)

    # run the tests
    test_counter = 0
    for name, command in UNITTESTS:
        test_counter += run_test(name, command)
    print(colored('{}/{} TESTS are OK'.format(test_counter, len(UNITTESTS)), 'cyan'))

if __name__ == "__main__":
    main()
