from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env = gym.make('Stochastic-8x8-FrozenLake-v0')
    env.render()

    input('Hit enter to run value iteration...')
    start = time.time()
    values, num_value_iters = value_iteration(env, gamma)
    end = time.time()
    print("Execute time", end - start)
    print("The number of value iterations: %d" % num_value_iters)


if __name__ == '__main__':
    main()

