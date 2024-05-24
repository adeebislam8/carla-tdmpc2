from stable_baselines3.common.env_checker import check_env
from carla_env.env import CarlaEnv

env=CarlaEnv()
check_env(env,warn=True)