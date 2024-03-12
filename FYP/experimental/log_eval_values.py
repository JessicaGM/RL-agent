from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from FYP.agent_components.config_env import ConfigEnv

action_type = "high-level"
model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
eval_path = "eval_logs/highway-env_20-cars_PPO_high-level"
custom_rewards = "no"

# action_type = "continuous"
# model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
# eval_path = "eval_logs/highway-env_20-cars_PPO_continuous"
# custom_rewards = "no"

env = ConfigEnv().create(action_type=action_type, custom_rewards=custom_rewards)
model = PPO.load(model_path)

if action_type == "high-level":
    env_c = Monitor(env, eval_path, info_keywords=('crashed', 'speed', 'HL_step_count', 'LL_step_count', 'pos_x',
                                                   'pos_y'))
if action_type == "continuous":
    env_c = Monitor(env, eval_path, info_keywords=('crashed', 'speed', 'step_count', 'pos_x', 'pos_y'))
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
