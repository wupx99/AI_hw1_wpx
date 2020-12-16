from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from rl_agent import *


def main():
    gamma = 0.9
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    env = gym.make('Deterministic-8x8-FrozenLake-v0')
    env.render()

    input('Hit enter to run value iteration...')
    values, num_value_iters = value_iteration(env, gamma)
    policy = value_function_to_policy(env, gamma, values)
    print_values(values)
    reward, step = run_policy(env, gamma, policy)
    print("total cumulative discounted reward: ", reward)


if __name__ == '__main__':
    main()
