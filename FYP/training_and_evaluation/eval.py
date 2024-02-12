import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from FYP.config_env import ConfigEnv
from FYP.custom_wrapper import CustomWrapper

env = ConfigEnv().make_configured_env
env = env()
#env = CustomWrapper(env)

algorithm_type = "PPO"
model_path = "models/highway-env_0-cars_PPO_continuous/model.zip"
eval_path = "eval_logs/highway-env_0-cars_PPO_continuous"

if algorithm_type == "PPO":
    model = PPO.load(model_path)
elif algorithm_type == "TQC":
    model = TQC.load(model_path)
else:
    raise ValueError("Specify a valid algorithm type.")

env_c = Monitor(env, eval_path, info_keywords=('distance_covered',))
env_c.reset()

# Number of episodes for evaluation
num_episodes = 100

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
