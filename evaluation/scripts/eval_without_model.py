import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy

env = gym.make("highway-v0", render_mode="human")

model = PPO(MlpPolicy, env, verbose=0)

env_c = Monitor(env, '../logs/eval_without_model', info_keywords=('distance_covered',))
env_c.reset()

# Number of episodes for evaluation
num_episodes = 100

for episode in range(num_episodes):
    # print("Episode:", episode + 1)
    done = truncated = False
    obs, info = env_c.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_c.step(action)

        # Add custom metric to the monitor log csv
        distance_covered = info.get("distance_covered", 0.0)
        env_c.render()

        # print(info)

    env_c.reset()
