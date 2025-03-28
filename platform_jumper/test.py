import gymnasium as gym
from platform_jumper_env import PlatformJumperEnv

env = PlatformJumperEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
