import gymnasium as gym


class ContinuousActions(gym.ActionWrapper):
    """
    A Gymnasium Action Wrapper to track certain events within an environment for continuous actions.

    Attributes:
        step_count (int): Counter for the number of steps taken in the current episode.
        timesteps (int): Total counter for steps taken across all episodes, useful for training.
        episode_count (int): Counter for the number of episodes processed, useful for training info.
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

    def reset(self, **kwargs):
        """
        Resets the environment and step counters.

        Returns:
           The initial env.
        """
        self.step_count = 0
        self.episode_count += 1

        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Takes an action in the environment and updates counters based on the result.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the observation, reward, terminated, truncated, and info flags.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        self.timesteps += 1

        # useful for evaluation
        info["step_count"] = self.step_count    # should be the same as episode len
        info["pos_x"] = obs[0][1]
        info["pos_y"] = obs[0][2]

        return obs, reward, terminated, truncated, info
