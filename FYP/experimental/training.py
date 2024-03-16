from stable_baselines3 import PPO
from FYP.agent_components.config_env import ConfigEnv
from FYP.experimental.custom_callback import CustomCallback


class Training:
    """
    Class for training.

    Attributes:
        env: The environment for the agent.
        action_type (str): The type of action for the environment ('continuous' or 'high-level' for continuous
            or hierarchical agent, respectively).
        log_path (str): The directory path where log files will be saved.
        model_path (str): The file path for saving the trained or updated model.
        total_timesteps (int): The total number of timesteps for training.
        mode (str): The mode of operation ('train' or 'train_more' to train new model or to continue training a model,
            respectively).
    """

    def __init__(self, env, action_type, log_path, model_path, total_timesteps=int(2e4), mode="train"):
        self.env = env
        self.action_type = action_type
        self.log_path = log_path
        self.model_path = model_path
        self.total_timesteps = total_timesteps
        self.mode = mode

    def run(self):
        """
        Trains or continues training an agent on the provided environment.

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

        if self.mode == "train":
            model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, n_steps=n_steps,
                        batch_size=batch_size, n_epochs=10, learning_rate=5e-4,
                        gamma=0.8, verbose=2, tensorboard_log=self.log_path)
        else:
            model = PPO.load(self.model_path, env=self.env)

        callback = CustomCallback(env_wrapper=self.env, action_type=self.action_type, verbose=1)
        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        # Save the model after training
        model.save(self.model_path if self.mode == "train" else self.model_path.replace("model", "updated_model"))

        return model


if __name__ == "__main__":
    # Configuration

    action_type = "high-level"
    mode = "train"
    log_path = "models/highway-env_20-cars_PPO_high-level/"
    model_path = log_path + "model.zip"
    custom_rewards = "no"
    render_mode = "human"

    # action_type = "continuous"
    # mode = "train"
    # log_path = "models/highway-env_20-cars_PPO_continuous/"
    # model_path = log_path + "model.zip"
    # custom_rewards = "no"
    # render_mode = "human"

    # Environment setup
    env = ConfigEnv().create(action_type=action_type, render_mode=render_mode, custom_rewards=custom_rewards)

    # Train or continue training the model
    model = Training(env=env, action_type=action_type, log_path=log_path, model_path=model_path, mode=mode).run()
