import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make("highway-v0", render_mode="human")

model = PPO.load("highway_ppo/model")

env_c = Monitor(env, 'logs/eval_with_model', info_keywords=('distance_covered',))
env_c.reset()

# Number of episodes for evaluation
num_episodes = 1000

all_episode_returns = []  # rewards
all_episode_distance_covered = []
all_episode_lengths = []
all_episode_times = []
for episode in range(num_episodes):
    # print("Episode:", episode + 1)
    done = truncated = False
    obs, info = env_c.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_c.step(action)

        # Add custom metric to the monitor log csv
        distance_covered = info.get("distance_covered", 0.0)
        all_episode_distance_covered.append(distance_covered)
        env_c.render()

        # print(info)

    all_episode_returns.append(sum(env_c.episode_returns))
    all_episode_lengths.append(sum(env_c.episode_lengths))
    all_episode_times.append(sum(env_c.episode_times))
    env_c.reset()

mean_episode_reward = np.mean(all_episode_returns)
mean_episode_distance_covered = np.mean(all_episode_distance_covered)
mean_episode_length = np.mean(all_episode_lengths)
mean_episode_time = np.mean(all_episode_times)

# print("Num episodes:", num_episodes)
# print("Mean reward:", mean_episode_reward)
# print("Mean distance covered:", mean_episode_distance_covered)
# print("Mean episode length:", mean_episode_length)
# print("Mean episode time:", mean_episode_time)
