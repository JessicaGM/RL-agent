import gymnasium as gym
from matplotlib import pyplot as plt

from FYP.agent_components.config_env import ConfigEnv


env = ConfigEnv().create(render_mode="human", action_type="continuous", custom_rewards="no")
# env = ConfigEnv().create(render_mode="human", action_type="high-level", custom_rewards="no")

done = truncated = False
obs, info = env.reset()
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()
