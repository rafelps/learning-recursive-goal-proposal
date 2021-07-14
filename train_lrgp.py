import os
import argparse
import gym
import gym_simple_minigrid  # just to register envs
import numpy as np
import torch

from lrgp.hierarchy import Hierarchy


parser = argparse.ArgumentParser()

parser.add_argument('--job_name', type=str, required=True, help='Name to identify this training session')
parser.add_argument('--env', type=str, default='Simple-MiniGrid-FourRooms-15x15-v0', help='Environment to use')
parser.add_argument('--seed', type=int, default=12345, help='Seed to control the training process')
parser.add_argument('--n_episodes', type=int, default=25000, help='Number of episodes')
parser.add_argument('--test_each', type=int, default=50, help='Include testing episodes each training episodes')
parser.add_argument('--n_episodes_test', type=int, default=50, help='Number of test episodes to average results')
parser.add_argument('--update_each', type=int, default=1, help='Update NNs each training episodes')
parser.add_argument('--n_updates', type=int, default=5, help='Number of NNs updates after each episode')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--low_h', type=int, default=4, help='Low horizon: maximum number of steps the low agent can do '
                                                         'every time it is used')
parser.add_argument('--high_h', type=int, default=15, help='High horizon: maximum number of subgoal proposals per '
                                                           'episode')
parser.add_argument('--epsilon_max', type=float, default=0.65, help='Maximum exploration probability for e-greedy '
                                                                    'policy')
parser.add_argument('--epsilon_min', type=float, default=0.1, help='Minimum exploration probability for e-greedy '
                                                                    'policy')
parser.add_argument('--epsilon_decay', type=float, default=0.9994, help='Decay for epsilon')

args = parser.parse_args()
args = vars(args)

# Make env
env = gym.make(args['env'])

# Seed everything
env.seed(args['seed'])
env.action_space.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

# Create epsilon exploration curve
epsilon_f = lambda i: args['epsilon_min'] + (args['epsilon_max'] - args['epsilon_min']) * args['epsilon_decay'] ** i

# Train
print(f"Running {args['job_name']}...")
learner = Hierarchy(env)
learner.train(**args, epsilon_f=epsilon_f)

# Save checkpoints and logs
learner.save(os.path.join('logs', args['job_name']))
