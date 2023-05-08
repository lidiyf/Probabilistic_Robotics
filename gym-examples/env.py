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
env = make_vec_env('gym_examples/GridWorld-v0', n_envs=10, monitor_dir="mon/")
#env = Monitor(env, filename="result")

#model = PPO(MultiInputActorCriticPolicy, env, verbose=1)
model = A2C(MultiInputActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=1000000, progress_bar=True)
model.save("a2c")

'''
observation, info = env.reset()
while True: #for _ in range(10000):
    #action = env.action_space.sample()
    action, _states = model.predict(observation)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation['goal'])

    if terminated or truncated:
        observation, info = env.reset()

    env.render()
#env.close()
'''