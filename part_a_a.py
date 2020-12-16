from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # env = gym.make('Deterministic-8x8-FrozenLake-v0')
    env.render()

    input('Hit enter to run policy iteration...')
    start = time.time()
    policy, value_func, num_policy_imp, num_value_iters = policy_iteration(env, gamma)
    end = time.time()
    print("Execute time", end - start)
    print("The number of policy improvements: %d" % num_policy_imp)
    print("The number of value iterations: %d" % num_value_iters)


if __name__ == '__main__':
    main()
