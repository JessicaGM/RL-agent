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
        # Initialize reward components to None; they will be updated during each step.
        self.collision_reward = None
        self.right_lane_reward = None
        self.high_speed_reward = None

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """
        Modifies the reward based on defined criteria.

        Parameters:
        reward (SupportsFloat): The original reward from the environment.

        Returns:
        SupportsFloat: The modified reward.
        """
        self._calculate_reward_components()
        reward = self._compute_final_reward()

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
            'collision_reward': self.collision_reward,
            'right_lane_reward': self.right_lane_reward,
            'high_speed_reward': self.high_speed_reward
        }

        return obs, reward, done, truncated, info

    def _calculate_reward_components(self):
        """
        Calculates individual components of the reward based on the vehicle's state.
        """
        vehicle = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road
        config = self.env.unwrapped.config

        self.collision_reward = 0 if not vehicle.crashed else float(config["collision_reward"])
        self.on_road_reward = float(vehicle.on_road)

        self._calculate_right_lane_reward(vehicle, road)
        self._calculate_high_speed_reward(vehicle, config)

    def _calculate_right_lane_reward(self, vehicle, road):
        """
        Calculates the reward for being in the right lane and being centered in the lane.

        Parameters:
        vehicle: The vehicle for which the reward is being calculated.
        road: The road on which the vehicle is located.
        """
        self.right_lane_reward = 0  # Default to 0
        if 0.1 > vehicle.lane_offset[1] > -0.1:
            lane_position = vehicle.lane_index[2]
            neighbours = road.network.all_side_lanes(vehicle.lane_index)
            self.right_lane_reward = lane_position / max(len(neighbours) - 1, 1)

    def _calculate_high_speed_reward(self, vehicle, config):
        """
        Calculates the reward for the vehicle's speed, scaling it within a specified range.

        Parameters:
        vehicle: The vehicle for which the reward is being calculated.
        config: The environment configuration.
        """
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, config["reward_speed_range"], [0, 1])
        self.high_speed_reward = np.clip(scaled_speed, 0, 1)

    def _compute_final_reward(self) -> SupportsFloat:
        """
        Computes the final reward by summing up the individual reward components.

        Returns:
        SupportsFloat: The final computed reward.
        """
        config = self.env.unwrapped.config
        reward = (self.collision_reward + self.right_lane_reward + self.high_speed_reward)

        # Normalize the reward if specified in the configuration
        if config.get("normalize_reward", False):
            reward = utils.lmap(reward,
                                [config["collision_reward"], config["high_speed_reward"] + config["right_lane_reward"]],
                                [0, 1])
        reward *= self.on_road_reward

        return reward
