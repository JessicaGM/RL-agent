from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from FYP.config_env import ConfigEnv
from FYP.custom_wrapper import CustomWrapper


class CustomCallback(BaseCallback):
    def __init__(self, env_wrapper, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env_wrapper = env_wrapper

    def _on_step(self) -> bool:
        # Log the low-level step count to TensorBoard
        if self.env_wrapper is not None:
            self.logger.record("other/LL_step_count", self.env_wrapper.LL_step_count)
        return True  # Continue training


def train_ppo(env, env_action_type):
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
    if env_action_type == "high-level":
        model.learn(total_timesteps=int(5e4), callback=CustomCallback(env_wrapper=env, verbose=1))
    else:
        model.learn(total_timesteps=int(5e4))

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
        model = train_ppo(env, env_action_type)

    if mode == "train_more":
        # Load the initial model for further training
        model = PPO.load(model_path, env=env)

        # Continue training the loaded model for a longer duration
        if env_action_type == "high-level":
            model.learn(total_timesteps=int(5e4), callback=CustomCallback(env_wrapper=env, verbose=1))
        else:
            model.learn(total_timesteps=int(5e4))

        # Save the model after additional training
        model.save(updated_model_path)
