import gymnasium
from typing import SupportsFloat, Any
import numpy as np
from highway_env import utils


class CustomReward(gymnasium.RewardWrapper):
    """
    A custom reward wrapper for environments that modifies the reward based on specific criteria such as
    collision avoidance, lane adherence, speed.

    Adapted from: highway_env.py
    """

    def __init__(self, env):
        """
        Initializes the reward wrapper.

        Parameters:
        env (gym.Env): The environment to wrap.
        """
        super().__init__(env)
        self.right_lane_reward = 0
        self.high_speed_reward = 0
        self.lane_position_tolerance = (1 / (self.env.unwrapped.config['policy_frequency'])) * 1.5

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """
        Modifies the reward based on defined criteria.

        Parameters:
        reward (SupportsFloat): The original reward from the environment.

        Returns:
        SupportsFloat: The modified reward.
        """
        vehicle = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road
        config = self.env.unwrapped.config

        self.calculate_reward_components(vehicle, road, config)
        reward = self.compute_final_reward(vehicle, config)

        return reward

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment, applying the action.

        Parameters:
        action: The action to apply in the environment.

        Returns:
        A tuple containing observation, modified reward, done flag, truncated flag, and info dict.
        """
        obs, reward, done, truncated, info = super().step(action)
        reward = self.reward(reward)  # Modify the reward based on custom logic

        # Update the 'info' dictionary with detailed reward components
        info['rewards'] = {
            'right_lane_reward': self.right_lane_reward,
            'high_speed_reward': self.high_speed_reward
        }

        info['on_road'] = self.env.unwrapped.vehicle.on_road

        return obs, reward, done, truncated, info

    def calculate_reward_components(self, vehicle, road, config):
        """
        Calculates individual components of the reward based on the vehicle's state.
        """

        self.calculate_lane_reward(vehicle, road)
        self.calculate_speed_reward(vehicle, config)

    def calculate_lane_reward(self, vehicle, road):
        """
        Calculates the reward for being in the right lane and being centered in the lane.

        Parameters:
        vehicle: The vehicle for which the reward is being calculated.
        road: The road on which the vehicle is located.
        """
        lane_position = vehicle.lane_index[2]
        neighbours = road.network.all_side_lanes(vehicle.lane_index)
        if self.lane_position_tolerance > vehicle.lane_offset[1] > - self.lane_position_tolerance:
            self.right_lane_reward = lane_position / max(len(neighbours) - 1, 1)
        else:
            self.right_lane_reward = 0

    def calculate_speed_reward(self, vehicle, config):
        """
        Calculates the reward for the vehicle's speed, scaling it within a specified range.

        Parameters:
        vehicle: The vehicle for which the reward is being calculated.
        config: The environment configuration.
        """
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, config["reward_speed_range"], [0, 1])
        if scaled_speed > 1:
            self.high_speed_reward = 0
        else:
            self.high_speed_reward = np.clip(scaled_speed, 0, 1)

    def compute_final_reward(self, vehicle, config) -> SupportsFloat:
        """
        Computes the final reward by summing up the individual reward components.

        Returns:
        SupportsFloat: The final computed reward.
        """
        reward = (config["right_lane_reward"] * self.right_lane_reward
                  + config["high_speed_reward"] * self.high_speed_reward)

        # Normalize the reward if specified in the configuration
        if config.get("normalize_reward", False):
            reward = utils.lmap(reward,
                                [0, config["high_speed_reward"] + config["right_lane_reward"]],
                                [0, 1])
        reward *= float(vehicle.on_road)
        reward *= float(not vehicle.crashed)
        return reward
