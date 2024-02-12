import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("highway-v0", render_mode="human")

model = PPO.load("../../other/highway_ppo_c_no/model")

env_c = Monitor(env, '../logs/eval_with_PPO_c_no_model', info_keywords=('distance_covered',))
env_c.reset()

# Number of episodes for evaluation
num_episodes = 1000

for episode in range(num_episodes):
    print("Episode:", episode + 1)
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
