import logging

from stable_baselines3.common.monitor import Monitor

from FYP.agent_components.config_env import ConfigEnv
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LoadModel:
    """Class for loading trained model to interact with the environment."""

    def __init__(self, model_path, action_type, custom_rewards, render_mode=None, algorithm_type="PPO",
                 eval_log_path=None,
                 num_episodes=1):
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
        - eval_log_path (str): Path to store evaluation log (optional). If specified Monitor wrapper will be used to
                    save the episode reward, length, time and other data for specified number of episodes in a csv file.
        - num_episodes (int): Number of episodes to render or perform evaluation. Default to 1.
        """
        self.model_path = model_path
        self.algorithm_type = algorithm_type
        self.model = None
        self.action_type = action_type
        self.custom_rewards = custom_rewards
        self.render_mode = render_mode
        self.eval_log_path = eval_log_path
        self.num_episodes = num_episodes

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

        if self.eval_log_path is not None:
            if self.action_type == "high-level":
                info_keywords = ('crashed', 'speed', 'HL_step_count', 'LL_step_count', 'pos_x', 'pos_y')
            elif self.action_type == "continuous":
                info_keywords = ('crashed', 'speed', 'step_count', 'pos_x', 'pos_y')
            else:
                logging.error("Unsupported action type")
                return

            env = Monitor(env, self.eval_log_path, info_keywords=info_keywords)

        # Load the model
        try:
            self.load_model()
        except Exception as e:
            logging.error(f"An error occurred during loading the model: {e}")

        for episode in range(self.num_episodes):
            print("Episode:", episode + 1)
            done = truncated = False
            obs, info = env.reset()

            while not (done or truncated):
                # Predict the action using the model
                action, _ = self.model.predict(obs, deterministic=True)
                # Take the predicted action in the environment
                obs, reward, done, truncated, info = env.step(action)

        env.close()


if __name__ == "__main__":
    # render one
    model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
    action_type = "high-level"
    custom_rewards = "no"
    render_mode = "human"

    # model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
    # action_type = "continuous"
    # custom_rewards = "no"
    # render_mode = "human"

    LoadModel(model_path=model_path, action_type=action_type, custom_rewards=custom_rewards,
              render_mode=render_mode).interact_with_environment()

    # to evaluate multiple episodes
    # model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
    # action_type = "high-level"
    # custom_rewards = "no"
    # eval_log_path = "eval_logs/highway-env_20-cars_PPO_high-level"
    # num_episodes = 100

    # model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
    # action_type = "continuous"
    # custom_rewards = "no"
    # eval_log_path = "eval_logs/highway-env_20-cars_PPO_continuous"
    # num_episodes = 100

    # LoadModel(model_path=model_path, action_type=action_type, custom_rewards=custom_rewards,
    #           eval_log_path=eval_log_path, num_episodes=num_episodes).interact_with_environment()
