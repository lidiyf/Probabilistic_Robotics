import gym_examples
import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

env = gymnasium.make('gym_examples/GridWorld-v0', render_mode="human")
#model = PPO.load("ppo")
#model = A2C.load("a2c")

observation, info = env.reset()
for _ in range(10000): #while True: #
    action = env.action_space.sample()
    #action, _states = model.predict(observation)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    env.render()
env.close()