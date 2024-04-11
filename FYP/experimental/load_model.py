import logging

from stable_baselines3.common.monitor import Monitor

from FYP.agent_components.config_env import ConfigEnv
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LoadModel:
    """Class for loading trained model to interact with the environment."""

    def __init__(self, model_path, action_type, custom_rewards, render_mode=None, algorithm_type=None,
                 eval_log_path=None, num_episodes=1):
        """
        Initialize instance.

        Parameters:
        - model_path (str): Path to the saved RL model.
        - algorithm_type (str): Type of reinforcement learning algorithm used PPO or None, None by default -
            it result random action selection rather than use of trained model.
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
        elif self.algorithm_type is None:
            logging.info("No algorithm specified, using random actions")
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
                info_keywords = ('HL_step_count', 'LL_step_count', 'pos_x', 'pos_y', 'speed', 'average_speed',
                                 'right_lane_count', 'crashed', 'on_road', 'truncated', )
            elif self.action_type == "continuous":
                info_keywords = ('step_count', 'pos_x', 'pos_y', 'speed', 'average_speed', 'right_lane_count',
                                 'crashed', 'on_road', 'truncated', )
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
                if self.algorithm_type is None:
                    action = env.action_space.sample()
                else:
                    # Predict the action using the model
                    action, _ = self.model.predict(obs, deterministic=True)
                # Take the predicted action in the environment
                obs, reward, done, truncated, info = env.step(action)

        env.close()


if __name__ == "__main__":
    # See parameters in the top of this class to configure own render/evaluation

    # Examples:
    # to render/evaluate 1000 episodes for high-level with PPO model
    model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
    action_type = "high-level"
    algorithm_type = "PPO"
    custom_rewards = "no"
    render_mode = "human"
    eval_log_path = "eval_logs/highway-env_20-cars_PPO_high-level"
    num_episodes = 1000

    # to render/evaluate 1000 episodes for continuous with PPO model
    # model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
    # action_type = "continuous"
    # algorithm_type = "PPO"
    # custom_rewards = "no"
    # render_mode = "human"
    # eval_log_path = "eval_logs/highway-env_20-cars_PPO_continuous-tes"
    # num_episodes = 1000

    # to render/evaluate 1000 episodes for high-level without model
    # model_path = "models/highway-env_20-cars_no-model_high-level/model.zip"
    # action_type = "high-level"
    # algorithm_type = None
    # custom_rewards = "no"
    # eval_log_path = "eval_logs/highway-env_20-cars_no-model_high-level"
    # num_episodes = 1000

    # to render/evaluate multiple episodes for continuous without model
    # model_path = "models/highway-env_20-cars_no-model_continuous/model.zip"
    # action_type = "continuous"
    # algorithm_type = None
    # custom_rewards = "no"
    # eval_log_path = "eval_logs/highway-env_20-cars_no-model_continuous"
    # num_episodes = 1000

    LoadModel(model_path=model_path, algorithm_type=algorithm_type, action_type=action_type,
              custom_rewards=custom_rewards, render_mode=render_mode, eval_log_path=eval_log_path, num_episodes=num_episodes).interact_with_environment()
