import gymnasium as gym
# import gym_navigation

# env = gym.make('gym_navigation:NavigationTrack-v0',
#                render_mode='human',
#                track_id=1)
print('---init env---')
env = gym.make('gym_navigation:NavigationGoal-v0',
               render_mode='human',
               track_id=2)
print('---seed action space---')
env.action_space.seed(42)

print('---reset---')
observation, info = env.reset(seed=42)
print(f'observation={observation} info={info}')


for _ in range(500):
    print('---sample---')
    action = env.action_space.sample()
    print('---step---')
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'action={action} '
          f'observation={observation} '
          f'reward={reward} '
          f'terminated={terminated} '
          f'info={info}')

    if terminated:
        observation, info = env.reset()
        print(f'observation={observation} info={info}')

print('---close env---')
env.close()
