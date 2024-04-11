import gymnasium as gym


class ContinuousActions(gym.ActionWrapper):
    """
    A Gymnasium Action Wrapper to track additional metrics within an environment for continuous actions.

    Attributes:
        step_count (int): The number of steps taken in the current episode.
        timesteps (int): The cumulative number of steps taken across all episodes.
        episode_count (int): The number of episodes processed by the wrapper.
        total_speed (float): The total speed accumulated over the current episode, useful for calculating the average speed.
        rightmost_lane (int): The index of the rightmost lane in the environment, based on environment configuration.
        right_lane_count (int): The number of steps the agent spends in the rightmost lane during the current episode.
    """

    def __init__(self, env):
        """
        Initializes the custom wrapper.

        Args:
            env (gym.Env): The original Gymnasium environment to be wrapped.
        """
        super().__init__(env)
        self.step_count = 0
        self.timesteps = 0
        self.episode_count = 0
        self.total_speed = 0
        self.rightmost_lane = self.env.unwrapped.config['lanes_count'] - 1
        self.right_lane_count = 0

    def reset(self, **kwargs):
        """
        Resets the environment and step counters.

        Returns:
           The initial env.
        """
        self.step_count = 0
        self.episode_count += 1
        self.total_speed = 0
        self.right_lane_count = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Takes an action in the environment and updates counters based on the result.

        Args:
            action: The action to take.

        Returns:
            obs: The new observation after taking the action.
            reward: The reward resulting from the action.
            terminated: A boolean indicating if the episode has ended.
            truncated: A boolean indicating if the episode was truncated.
            info: A dictionary with additional information, including custom tracking metrics.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        self.timesteps += 1
        self.total_speed += self.env.unwrapped.vehicle.speed
        if self.env.unwrapped.vehicle.lane_index[2] == self.rightmost_lane:
            self.right_lane_count += 1

        # Additional information for evaluation purposes
        info["step_count"] = self.step_count    # should be the same as episode len
        info["pos_x"] = obs[0][1]
        info["pos_y"] = obs[0][2]
        info["average_speed"] = self.total_speed / max(1, self.step_count)
        info["right_lane_count"] = self.right_lane_count
        info["on_road"] = self.env.unwrapped.vehicle.on_road
        info["truncated"] = truncated

        return obs, reward, terminated, truncated, info
