class EnvConfig:
    def env_config(self):
        """
        Set the configuration for the environment.

        Returns:
            dict: A dictionary containing configuration parameters for the environment.

        The configuration includes settings for observation, action, lanes, vehicles,
        duration, rewards, screen dimensions, and other environmental parameters.
        """
        config = {
            "observation": {
                "type": "Kinematics",
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 5,
            "vehicles_count": 50,
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
            "offroad_terminal": True,
            "screen_width": 1000,
            "screen_height": 500,
            "centering_position": [0.1, 0.5],
            "longitudinal": True,
            "lateral": True,
            "order": "sorted"
        }

        return config
