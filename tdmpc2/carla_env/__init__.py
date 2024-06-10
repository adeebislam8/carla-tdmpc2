# from gymnasium.envs.registration import register
from gym.envs.registration import register

register(
     id="CarlaEnv-v0",
     entry_point="carla_env.env:CarlaEnv",
     max_episode_steps=1000,
)

register(
     id="CarlaEnv-v1",
     entry_point="carla_env.env_iso:CarlaEnv",
     # max_episode_steps=1000,
)