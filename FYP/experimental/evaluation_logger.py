from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from FYP.agent_components.config_env import ConfigEnv


class EvaluationLogger:
    """
    Class for logging values during evaluation.

    Attributes:
        action_type (str): Type of actions used in the environment.
        model_path (str): Path to the trained RL model.
        eval_path (str): Path to store evaluation logs.
        custom_rewards (str): Specifies whether custom rewards are used or not.
        num_episodes (int): Number of episodes for evaluation.
        env: The environment.
        model: The RL model.
    """
    def __init__(self, action_type, model_path, eval_path, custom_rewards, num_episodes=100):
        self.action_type = action_type
        self.model_path = model_path
        self.eval_path = eval_path
        self.custom_rewards = custom_rewards
        self.num_episodes = num_episodes
        self.env = None
        self.model = None

    def run(self):
        """
        Evaluate the RL model by interacting with the environment.

        This method sets up the environment, loads the trained model, and performs evaluation
        by running a specified number of episodes.
        """
        # Setup environment
        env = ConfigEnv().create(action_type=self.action_type, custom_rewards=self.custom_rewards)
        if self.action_type == "high-level":
            info_keywords = ('crashed', 'speed', 'HL_step_count', 'LL_step_count', 'pos_x', 'pos_y')
        elif self.action_type == "continuous":
            info_keywords = ('crashed', 'speed', 'step_count', 'pos_x', 'pos_y')
        else:
            raise ValueError("Unsupported action type")
        env = Monitor(env, self.eval_path, info_keywords=info_keywords)

        # Load model
        model = PPO.load(self.model_path)

        # Reset environment
        env.reset()

        for episode in range(self.num_episodes):
            print("Episode:", episode + 1)
            done = truncated = False
            obs, info = env.reset()

            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)


if __name__ == "__main__":
    action_type = "high-level"
    model_path = "models/highway-env_20-cars_PPO_high-level/model.zip"
    eval_path = "eval_logs/highway-env_20-cars_PPO_high-level"
    custom_rewards = "no"

    # action_type = "continuous"
    # model_path = "models/highway-env_20-cars_PPO_continuous/model.zip"
    # eval_path = "eval_logs/highway-env_20-cars_PPO_continuous"
    # custom_rewards = "no"

    EvaluationLogger(action_type, model_path, eval_path, custom_rewards).run()
