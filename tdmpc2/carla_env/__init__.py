# from gymnasium.envs.registration import register
from gym.envs.registration import register

register(
     id="CarlaEnv-v0",
     entry_point="carla_env.env:CarlaEnv",
     max_episode_steps=1000,
)