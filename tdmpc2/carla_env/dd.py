import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from carla_env.env import CarlaEnv

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CarlaEnv()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=rank)
        return env

    set_random_seed(seed)
    return _init


#env=DummyVecEnv([lambda: CarlaEnv()])
#env = make_vec_env(CarlaEnv, n_envs = 2, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
env = SubprocVecEnv([make_env(i) for i in range(8)], start_method="fork")

model = PPO(MlpPolicy, env, verbose=1, learning_rate=0.001)
#print("111111111")
model.learn(total_timesteps=1_000_000)

model.save("model"+"/PPO_model")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(50_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()