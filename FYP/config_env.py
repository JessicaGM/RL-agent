import gymnasium as gym


class ConfigEnv:
    """
    A utility class for configuring and creating a customized highway environment in Gymnasium.

    This class abstracts the setup process for the highway simulation, allowing for easy adjustments
    to environmental parameters such as the observation and action spaces, simulation dynamics, rewards,
    and rendering settings.

    Attributes:
        id (str): Identifier for the highway environment to be used with Gymnasium's make function.
        config (dict): A dictionary containing all configuration settings for the highway environment.
    """

    def __init__(self):
        """
        Initializes the ConfigEnv with a predefined set of configuration parameters for the highway environment.
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
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes,
                                       # linearly mapped to zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed,
                                       # linearly mapped to zero for lower speeds
                                       # according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,  # Terminate episode when offroad
            "screen_width": 1000,
            "screen_height": 500,
            "centering_position": [0.1, 0.5],
            "order": "sorted"
        }

    def make_configured_env(self, render_mode=None):
        """
        Instantiates and returns a configured highway environment based on the specified settings.

        Args:
            render_mode (str, optional): The render mode to be used by the environment. Defaults to None.

        Returns:
            gym.Env: The configured and initialized highway environment.
        """
        env = gym.make(self.id, render_mode=render_mode)
        env.configure(self.config)
        env.reset()
        return env
