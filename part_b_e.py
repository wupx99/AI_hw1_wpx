from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env = gym.make('Stochastic-8x8-FrozenLake-v0')

    values, num_value_iters = value_iteration(env, gamma)
    policy = value_function_to_policy(env, gamma, values)
    print_values(values)

    total_reward = 0
    for i in range(100):
        reward, step = run_policy(env, gamma, policy)
        total_reward += reward
    print("average reward: ", total_reward / 100)


if __name__ == '__main__':
    main()
