from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    action_names = lake_env.action_names
    env.render()

    values, num_value_iters = value_iteration(env, gamma)
    policy = value_function_to_policy(env, gamma, values)
    print_policy(policy, action_names)


if __name__ == '__main__':
    main()

