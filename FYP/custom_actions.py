import gymnasium as gym
from gymnasium import spaces

from FYP.lane_changer import LaneChanger
from FYP.speed_changer import SpeedChanger
from highway_env.vehicle.kinematics import Vehicle


class CustomActions(gym.ActionWrapper):
    """
    A custom actions for the Gymnasium environment to introduce high-level actions for an autonomous vehicle simulation.

    This wrapper modifies the original environment's action space to support high-level actions such as lane changing and
    speed adjustment, abstracting the lower-level actions required to perform these tasks. It keeps track of both
    high-level and low-level step counts to facilitate detailed analysis and monitoring during training.

    Attributes:
        env (gym.Env): The original Gymnasium environment being wrapped.
        HL_step_count (int): Counter for high-level steps taken in the environment.
        LL_step_count (int): Counter for low-level steps taken in the environment.
        action_space (spaces.Discrete): The action space of the environment after modification.
        leftmost_lane (int): Index of the leftmost lane.
        rightmost_lane (int): Index of the rightmost lane.

    Methods:
        reset(**kwargs): Resets the environment and counters.
        step(action): Performs a high-level action in the environment and returns the result.
    """

    def __init__(self, env):
        """
        Initializes the custom wrapper.

        Args:
            env (gym.Env): The original Gymnasium environment to be wrapped.
        """
        super().__init__(env)
        self.HL_step_count = 0
        self.LL_step_count = 0
        self.action_space = spaces.Discrete(5)  # Define new action space with 6 discrete actions
        self.leftmost_lane = 0
        self.rightmost_lane = self.env.unwrapped.config['lanes_count'] - 1

    def reset(self, **kwargs):
        """
        Resets the environment and step counters.

        Args:
            **kwargs: Additional arguments to the environment's reset method.

        Returns:
            The initial observation from the environment.
        """
        self.HL_step_count = 0
        self.LL_step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Performs a high-level action in the environment and updates step counts.

        This method abstracts away the specifics of performing high-level actions like lane changing or speed adjustment,
        handling the necessary low-level actions internally.

        Args:
            action (int): The high-level action to perform.
                - 0: Go forward for 10 meters.
                - 1: Change to the left lane.
                - 2: Change to the right lane.
                - 3: Slow down by around 1m/s.
                - 4: Speed up by around 1m/s.

        Returns:
            obs: The observation after taking the action.
            cumulative_reward: The reward obtained after executing the high-level action.
            terminated: A boolean indicating if the episode has ended.
            truncated: A boolean indicating if the episode was truncated.
            info: A dictionary with additional information about the step.
        """
        self.HL_step_count += 1

        terminated = False
        change = 0
        distance = None

        if 0 <= action <= 2:
            # Go forward around 10 meters
            if action == 0:
                change = 0
                distance = 10
            # Change left lane
            if action == 1:
                change = -1
            # Change right lane
            if action == 2:
                change = 1

            # Ensure that the vehicle is not on the rightmost or leftmost lane before executing the lane-changing action
            destination_lane = self.env.unwrapped.vehicle.lane_index[2] + change
            if self.leftmost_lane <= destination_lane <= self.rightmost_lane:
                changer = LaneChanger(self.env, change, distance)
            else:
                # If the desired lane exceeds the range of permissible lanes, stay on the current lane
                changer = LaneChanger(self.env, 0)

        elif 3 <= action <= 4:
            # Slow down by 1
            if action == 3:
                change = -1
            # Speed up by 1
            if action == 4:
                change = 1

            # Ensure that desired speed stays within the limit
            current_speed = self.env.unwrapped.vehicle.speed
            if Vehicle.MIN_SPEED <= current_speed + change <= Vehicle.MAX_SPEED:
                changer = SpeedChanger(self.env, change)
            else:
                # If desired speed exceeds the limit, just maintain current speed
                changer = SpeedChanger(self.env, 0)

        obs, reward, terminated, truncated, info = changer.step()
        cumulative_reward = reward
        while not changer.done() and not terminated:
            obs, reward, terminated, truncated, info = changer.step()
            cumulative_reward += reward

        self.LL_step_count += changer.step_count

        return obs, cumulative_reward, terminated, truncated, info
