import pandas as pd


class EvaluationProcessor:
    """
     A class to process evaluation metrics from CSV log files of simulation experiments.

     This class is designed to read simulation log data, compute, and display evaluation
     metrics relevant to RL AVs experiments. These metrics include the number of episodes, mean episode return,
     mean episode length, mean positional metrics, driving accuracy in terms of lane
     adherence, speed metrics, and incident rates such as collisions and off-road instances.

     Attributes:
         csv_file (str): The path to the CSV file containing the log data.
         action_type (str): The type of actions used in the simulation, e.g., `high-level` or `continuous`.
         num_episodes (int): The total number of episodes contained in the log.
         mean_episode_length (float): The average length of an episode, calculated differently based on the action type.
         mean_episode_return (float): The average return (reward) per episode.
         mean_pos_x (float): The average termination position (travelled distance) on the x-axis across episodes.
         mean_right_lane (float): The average percentage of steps the vehicle was in the right lane.
         mean_overall_speed (float): The average speed of the vehicle across episodes.
         collision_rate (float): The percentage of episodes that ended with a collision.
         off_road_rate (float): The percentage of episodes where the vehicle went off-road.
         truncated_rate (float): The percentage of episodes that were truncated.
     """

    def __init__(self, csv_file, action_type):
        self.csv_file = csv_file
        self.action_type = action_type
        self.num_episodes = None
        self.mean_episode_length = None
        self.mean_episode_return = None
        self.mean_pos_x = None
        self.mean_right_lane = None
        self.mean_overall_speed = None
        self.collision_rate = None
        self.off_road_rate = None
        self.truncated_rate = None

    def process_csv(self):
        """
        Processes the CSV file to calculate evaluation metrics.
        """
        try:
            df = pd.read_csv(self.csv_file, skiprows=1)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        self.num_episodes = len(df)
        self.mean_episode_return = df['r'].mean()

        if self.action_type == "high-level":
            self.mean_episode_length = df['LL_step_count'].mean()
        elif self.action_type == "continuous":
            self.mean_episode_length = df['l'].mean()

        self.mean_pos_x = df['pos_x'].mean()
        self.mean_right_lane = (df['right_lane_count'].mean() / self.mean_episode_length * 100
                                if self.mean_episode_length else 0)
        self.mean_overall_speed = df['average_speed'].mean()
        self.collision_rate = df['crashed'].astype(bool).mean() * 100
        self.off_road_rate = (~df['on_road'].astype(bool)).mean() * 100
        self.truncated_rate = df['truncated'].astype(bool).mean() * 100

    def display_results(self):
        """
        Displays the calculated evaluation metrics,
        providing insights into the performance of the simulated experiment of an agent.
        """
        print(f"Num episodes: {self.num_episodes}")
        print(f"Mean reward: {self.mean_episode_return:.2f}")
        print(f"Mean episode length: {self.mean_episode_length:.2f}")
        print(f"Mean pos x-axis: {self.mean_pos_x:.2f}")
        print(f"Mean right lane adherence: {self.mean_right_lane:.2f}%")
        print(f"Mean speed: {self.mean_overall_speed:.2f} m/s")
        print(f"Off-road rate: {self.off_road_rate:.2f}%")
        print(f"Collision rate: {self.collision_rate:.2f}%")
        print(f"Truncated rate: {self.truncated_rate:.2f}%")


if __name__ == "__main__":
    # Replace these

    action_type = "high-level"
    path = "eval_logs/highway-env_20-cars_PPO_high-level.monitor.csv"

    # action_type = "continuous"
    # path = "eval_logs/highway-env_20-cars_PPO_continuous.monitor.csv"

    evaluator = EvaluationProcessor(path, action_type)
    evaluator.process_csv()
    evaluator.display_results()
