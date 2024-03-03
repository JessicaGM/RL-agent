from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from FYP.config_env import ConfigEnv
from FYP.custom_actions import CustomActions


class CustomCallback(BaseCallback):
    """
    Custom callback for logging additional information during training.

    Attributes:
        env_wrapper (CustomActions): The environment wrapper to access additional data.
    """
    def __init__(self, env_wrapper, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env_wrapper = env_wrapper

    def _on_step(self) -> bool:
        """
        Logs the low-level step count to TensorBoard.

        Returns:
            bool: True to continue training, False otherwise.
        """
        if self.env_wrapper is not None:
            self.logger.record("other/LL_step_count", self.env_wrapper.LL_step_count)
        return True


def train_ppo(env, env_action_type, log_path, model_path, total_timesteps=int(2e4), mode="train"):
    """
    Trains or continues training a PPO agent on the provided environment.

    Args:
        env: The environment for the agent.
        env_action_type (str): The type of action for the environment ('continuous' or 'high-level' for continuous or hierarchical agent, respectively).
        log_path (str): The directory path where log files will be saved.
        model_path (str): The file path for saving the trained or updated model.
        total_timesteps (int): The total number of timesteps for training.
        mode (str): The mode of operation ('train' or 'train_more' to train new model or to continue training a model, respectively).

    Returns:
        The trained or updated PPO model.

    See:
    - `Parameters taken from: <https://github.com/Farama-Foundation/HighwayEnv/blob/a2497054390a5018060b1731bf643bcab3c53cd3/scripts/sb3_highway_ppo.py#L3>`
    """

    # Model configuration
    n_cpu = 6
    batch_size = 64
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    n_steps = batch_size * 12 // n_cpu

    if mode == "train":
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=n_steps,
                    batch_size=batch_size, n_epochs=10, learning_rate=5e-4,
                    gamma=0.8, verbose=2, tensorboard_log=log_path)
    else:
        model = PPO.load(model_path, env=env)

    callback = CustomCallback(env_wrapper=env, verbose=1) if env_action_type == "high-level" else None
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the model after training
    model.save(model_path if mode == "train" else model_path.replace("model", "updated_model"))

    return model


if __name__ == "__main__":
    # Configuration
    env_action_type = "continuous"
    mode = "train"
    log_path = "models/highway/"
    model_path = log_path + "model.zip"

    # Environment setup
    env = ConfigEnv().make_configured_env()
    env = CustomActions(env) if env_action_type == "high-level" else env

    # Train or continue training the model
    model = train_ppo(env, env_action_type, log_path, model_path, mode=mode)

