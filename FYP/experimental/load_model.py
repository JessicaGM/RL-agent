import logging
from FYP.agent_components.config_env import ConfigEnv
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LoadModel:
    """Class for loading trained model to interact with the environment."""

    def __init__(self, model_path, action_type, custom_rewards, render_mode, algorithm_type="PPO"):
        """
        Initialize instance.

        Parameters:
        - model_path (str): Path to the saved RL model.
        - algorithm_type (str): Type of reinforcement learning algorithm used (PPO only for now).
        - model: The loaded RL model, initially None until `load_model` is called.
        - action_type (str): Type of action space in the environment (high-level or continuous).
        - custom_reward (str): Specifies the type of rewards to be used in the environment. `no` for
                    original rewards from the original gym environment, `yes` for custom rewards.
        - render_mode (str): Render mode (`human`, `rgb_array`, or None) for visual output.
        """
        self.model_path = model_path
        self.algorithm_type = algorithm_type
        self.model = None
        self.action_type = action_type
        self.custom_rewards = custom_rewards
        self.render_mode = render_mode

    def load_model(self):
        """Load the RL model based on the specified algorithm type."""
        if self.algorithm_type == "PPO":
            try:
                self.model = PPO.load(self.model_path)
            except Exception as e:
                logging.error(f"Failed to load model: {e}")
                raise
        else:
            logging.error("Unsupported algorithm type")
            raise ValueError("Unsupported algorithm type")

    def interact_with_environment(self):
        """
        Interact with the gym environment using the loaded RL model.
        """
        try:
            env = ConfigEnv().create(action_type=self.action_type, custom_rewards=self.custom_rewards,
                                     render_mode=self.render_mode)
        except Exception as e:
            logging.error(f"Environment setup failed: {e}")
            return

        env.reset()

        # Load the model
        try:
            self.load_model()
        except Exception as e:
            logging.error(f"An error occurred during loading the model: {e}")

        # Reset the environment and get the initial observation
        obs, info = env.reset()

        done = False
        while not done:
            # Predict the action using the model
            action, _ = self.model.predict(obs)
            # Take the predicted action in the environment
            obs, reward, done, truncated, info = env.step(action)

        env.close()


if __name__ == "__main__":

    model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
    action_type = "high-level"
    custom_rewards = "no"
    render_mode = "human"

    # model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
    # action_type = "continuous"
    # custom_rewards = "no"
    # render_mode = "human"

    agent = LoadModel(model_path=model_path, action_type=action_type, custom_rewards=custom_rewards, render_mode=render_mode)
    agent.interact_with_environment()
