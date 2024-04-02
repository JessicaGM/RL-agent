import pandas as pd


class EvaluationProcessor:
    """
    Processes a CSV log file to calculate and display
    evaluation metrics such as the number of episodes, mean episode return,
    mean episode length etc.

    Attributes:
        csv_file (str): The path to the CSV file containing the log data.
        action_type (str): The type of actions used, e.g., `high-level` or `continuous`.
        num_episodes (int): The total number of episodes in the log.
        mean_episode_length (float): The average length of an episode
                    (for high-level action type is the total low-level steps).
        mean_episode_return (float): The average return per episode.
    """
    def __init__(self, csv_file, action_type):
        self.num_episodes = None
        self.mean_episode_length = None
        self.mean_episode_return = None
        self.csv_file = csv_file
        self.action_type = action_type

    def process_csv(self):
        """
        Processes the CSV file to calculate evaluation metrics. Calculates the total
        number of episodes, mean episode return, mean episode length etc.
        """
        df = pd.read_csv(self.csv_file, skiprows=1)
        self.num_episodes = len(df)
        self.mean_episode_return = df['r'].mean()
        if action_type == "high-level":
            self.mean_episode_length = df['LL_step_count'].mean()
        if action_type == "high-level":
            self.mean_episode_length = df['l'].mean()

    def display_results(self):
        """
        Displays the calculated evaluation metrics.
        """
        print(f"Num episodes: {self.num_episodes}")
        print(f"Mean reward: {self.mean_episode_return}")
        print(f"Mean episode length: {self.mean_episode_length}")


if __name__ == "__main__":
    action_type = "high-level"
    path = "eval_logs/highway-env_20-cars_PPO_high-level.monitor.csv"

    # action_type = "continuous"
    # path = "eval_logs/highway-env_20-cars_PPO_continuous.monitor.csv"

    evaluator = EvaluationProcessor(path, action_type)
    evaluator.process_csv()
    evaluator.display_results()
