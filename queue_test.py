#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
from builtins import input

import gym

import AI_hw1.queue_envs as queue_envs

if __name__ == "__main__":
    env = gym.make('Queue-1-v0')
    # print(env.query_model((0, 0, 0, 0), 1))
    env.step(2)
    for i in range(100):
        env.step(3)
        env.render()
