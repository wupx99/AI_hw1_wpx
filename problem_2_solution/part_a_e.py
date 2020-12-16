from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # env = gym.make('Deterministic-8x8-FrozenLake-v0')
    env.render()

    input('Hit enter to run value iteration...')
    values, num_value_iters = value_iteration(env, gamma)
    plot_values(values)


if __name__ == '__main__':
    main()
