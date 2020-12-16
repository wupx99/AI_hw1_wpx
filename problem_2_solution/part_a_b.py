from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # env = gym.make('Deterministic-8x8-FrozenLake-v0')
    action_names = lake_env.action_names
    env.render()

    input('Hit enter to run policy iteration...')
    policy, value_func, num_policy_imp, num_value_iters = policy_iteration(env, gamma)
    print_policy(policy, action_names)


if __name__ == '__main__':
    main()
