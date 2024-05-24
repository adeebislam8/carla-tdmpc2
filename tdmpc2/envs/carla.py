import os
import sys

import numpy as np
# import gymnasium as gym
import gym
# from gym.wrappers import TimeLimit
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.time_limit import TimeLimit
# ADB: Doesn't import preexisting library. Suitable to follow for CARLA environment.



def make_env(cfg):
    """
    Make Carla environment.
    """
    if not cfg.task.startswith("carla_"):
        raise ValueError("Unknown task:", cfg.task)
    import carla_env


    env = gym.make("CarlaEnv-v0", render_mode="human")
    print("env", env)
    env.max_episode_steps = 1000
    env = TimeLimit(env, max_episode_steps=1000)
    print("env", env)
    env = PixelWrapper(cfg, env, num_frames=3, render_size=64)
    print("env", env)
    # env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    print("Returning Carla environment.")
    return env
