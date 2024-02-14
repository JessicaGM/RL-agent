import gymnasium as gym
from gymnasium import spaces

from FYP.lane_changer import LaneChanger
from FYP.speed_changer import SpeedChanger


class CustomWrapper(gym.ActionWrapper):
    """
    A custom wrapper for modifying the behavior of an environment.

    This wrapper extends the gym.Wrapper class and is designed to modify the behavior of the underlying environment.

    Attributes:
        env (HighwayEnv): The original highway environment.

    Methods:
        reset(**kwargs):
            Reset the environment.

        step(action):
            Take a step in the environment based on the provided action.

    Usage:
        env = gym.make("highway-v0", render_mode="human")
        env = CustomWrapper(env)
    """

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.action_space = spaces.Discrete(6)

    def reset(self, **kwargs):
        """Reset the environment."""
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Take a step in the environment based on the provided action.

        Args:
            action (int): The action to take. Choose between:
                - 0: Go forward.
                - 1: Change to the left lane.
                - 2: Change to the right lane.
                - 3: Slow down by around 1m/s (gradually slow down).
                - 4: Speed up by around 1m/s (gradually speed up).
                - 5: Maintain the current speed.
        Returns:
            Tuple: A tuple containing the observation, reward, termination flag, truncated flag, and additional info.
        """
        self.step_count += 1
        # print("Step count:", self.step_count)

        terminated = False
        change = 0

        # Forward or Change lane
        if 0 <= action <= 2:
            # Go forward
            if action == 0:
                change = 0
            # Change left lane
            if action == 1:
                change = -1
            # Change right lane
            if action == 2:
                change = 1

            # Low-level
            changer = LaneChanger(self.env, change)
            obs, reward, terminated, truncated, info = changer.step()
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
            return obs, reward, terminated, truncated, info

        # Adjust speed
        if 3 <= action <= 5:
            # Gradually slow down
            if action == 3:
                change = -1  # change speed by -1
            # Gradually speed up
            if action == 4:
                change = 1  # change speed by 1
            # Maintain speed
            if action == 5:
                change = 0  # no change

            # Low-level
            changer = SpeedChanger(self.env, change)
            obs, reward, terminated, truncated, info = changer.step()
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
            return obs, reward, terminated, truncated, info
