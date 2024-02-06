import gymnasium as gym

from lane_changer import LaneChanger
from speed_changer import SpeedChanger


class CustomWrapper(gym.Wrapper):
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

    def reset(self, **kwargs):
        """Reset the environment."""
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Take a step in the environment based on the provided action."""
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
            changer = LaneChanger(self.env, change)
            obs, reward, terminated, truncated, info = changer.step()
            # print(obs, info)
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
                # print(obs, info)
            return obs, reward, terminated, truncated, info

        # Adjust speed
        if 3 <= action <= 6:
            # Decelerate
            if action == 3:
                change = -1  # change speed by -1
            # Accelerate
            if action == 4:
                change = 1  # change speed by 1
            # Maintain speed
            if action == 5:
                change = 0  # no change
            # Decelerate quicker
            changer = SpeedChanger(self.env, change)
            # print(obs, info)
            obs, reward, terminated, truncated, info = changer.step()
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
                # print(obs, info)
            return obs, reward, terminated, truncated, info
