import gymnasium as gym
from gymnasium.wrappers import record_video
from FYP.config_env import ConfigEnv

# Create the highway environment with rendering in RGB array mode
env = ConfigEnv().make_configured_env(render_mode="rgb_array")

env.reset()

video_folder = "videos/"
env = record_video.RecordVideo(env, video_folder=video_folder)

# Run the environment for some steps to record the video
done = False
while not done:
    action = env.action_space.sample()  # Update actions
    obs, reward, done, truncated, info = env.step(action)

# Close the environment to save the recorded video
env.close()
