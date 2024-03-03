import gymnasium as gym
import numpy as np


class ConfigEnv:
    """
    ConfigEnv class for configuring and creating a highway environment.

    The configuration includes settings for observation, action, lanes, vehicles,
    duration, rewards, screen dimensions, and other environmental parameters.

    Attributes:
        id (str): The ID of the highway environment.
        config (dict): Configuration parameters for the highway environment.
    """

    def __init__(self):
        """
        Initialize ConfigEnv with default ID and configuration parameters.
        """
        self.id = 'highway-v0'
        self.config = {
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "features": ["presence", "x", "y", "vx", "vy", "heading", "cos_h",
                             "sin_h", "cos_d", "sin_d", "long_off", "lat_off", "ang_off"],
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 15,  # [Hz]
            "lanes_count": 5,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,  # terminate episode
            "screen_width": 1000,
            "screen_height": 500,
            "centering_position": [0.1, 0.5],
            "order": "sorted"
        }

    def make_configured_env(self, render_mode=None):
        """
        Create and configure a highway environment based on the stored ID and configuration.

        Returns:
            gym.Env: Configured highway environment.
        """
        env = gym.make(self.id, render_mode=render_mode)
        env.configure(self.config)
        env.reset()
        return env