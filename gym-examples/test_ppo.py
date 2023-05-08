import gym_examples
import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

env = gymnasium.make('gym_examples/GridWorld-v0', render_mode="human")
model = PPO.load("ppo")

observation, info = env.reset()
while True: #for _ in range(10000):
    #action = env.action_space.sample()
    action, _states = model.predict(observation)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation['goal'])

    if terminated or truncated:
        observation, info = env.reset()

    env.render()