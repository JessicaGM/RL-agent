import sys
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
from FYP.config_env import ConfigEnv
from FYP.custom_wrapper import CustomWrapper


def train_ppo(env):
    """
    Train PPO agent on the provided environment.

    Args:
        env: The environment to train the agent on.

    Returns:
        PPO: The trained PPO model.

    See:
    - `Parameters taken from: <https://github.com/Farama-Foundation/HighwayEnv/blob/a2497054390a5018060b1731bf643bcab3c53cd3/scripts/sb3_highway_ppo.py#L3>`
    """
    n_cpu = 6
    batch_size = 64
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=2,
                tensorboard_log=log_path)

    # Train the agent
    model.learn(total_timesteps=int(5e4))

    # Save the model
    model.save(model_path)

    return model


def train_tqc(env):
    """
    Train TQC agent on the provided environment.

    Args:
        env: The environment to train the agent on.

    Returns:
        TQC: The trained TQC model.

    See:
    - `Parameters taken from: <https://github.com/Farama-Foundation/HighwayEnv/issues/331>`
    """
    model = TQC("MlpPolicy",
                env,
                policy_kwargs=dict(
                    n_critics=5,
                    n_quantiles=25,
                    net_arch=dict(pi=[256, 256], qf=[512, 512, 512]),
                    log_std_init=-3,
                ),
                batch_size=256,
                learning_rate=3e-4,
                train_freq=8,
                gradient_steps=8,
                tau=0.005,
                gamma=0.95,
                verbose=2,
                learning_starts=100,
                use_sde=True,
                tensorboard_log=log_path)

    # Train the agent
    model.learn(total_timesteps=int(2e4))

    # Save the model
    model.save(model_path)

    return model


if __name__ == "__main__":

    env_action_type = "high-level"
    """
    str: Specifies the type of action for the environment. 
    Choose between 'continuous' for default continuous agent 
    or 'high-level' for hierarchical agent.
    """

    algorithm_type = "PPO"
    """
    str: Specifies the reinforcement learning algorithm to be used.
    Choose between 'PPO' or 'TQC'.
    """

    mode = "train"
    """
    str: Specifies the mode of operation for the script. 
    Choose between 'train' to train a new model, or
    'train_more' to continue training a pre-trained model.
    """

    log_path = "models/highway"
    """
    str: Specifies the directory path where log files will be saved.
    """

    model_path = log_path + "model"
    """
    str: Specifies the file path for saving the trained model.
    """

    updated_model_path = log_path + "updated_model"
    """
    str: Specifies the file path for saving the updated model checkpoint after additional training.
    """

    env = ConfigEnv().make_configured_env()

    if env_action_type == "high-level":
        env = CustomWrapper(env)

    if mode == "train":
        if algorithm_type == "PPO":
            model = train_ppo(env)
        elif algorithm_type == "TQC":
            model = train_tqc(env)
        else:
            print("Specify a valid algorithm type.")

    if mode == "train_more":
        # Load the initial model for further training
        if algorithm_type == "PPO":
            model = PPO.load(model_path, env=env)
        elif algorithm_type == "TQC":
            model = TQC.load(model_path, env=env)
        else:
            print("Specify a valid algorithm type.")

        # Continue training the loaded model for a longer duration
        model.learn(total_timesteps=int(5e4))

        # Save the model after additional training
        model.save(updated_model_path)
