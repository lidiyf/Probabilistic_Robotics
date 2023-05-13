import gym_examples
import sys
import numpy as np
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from tqdm.rich import trange, tqdm

env = gymnasium.make('gym_examples/GridWorld-v0', render_mode="human")
#env = make_vec_env('gym_examples/GridWorld-v0', n_envs=2, monitor_dir="mon/")
env = Monitor(env, filename="result")

model = PPO(MultiInputActorCriticPolicy, env, verbose=1)
#model = A2C(MultiInputActorCriticPolicy, env, verbose=1)

model.learn(total_timesteps=800000, progress_bar=True)
model.save("ppo")
#model.save("a2c")