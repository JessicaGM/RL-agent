import gymnasium as gym

from FYP.agent_components.custom_reward import CustomReward
from FYP.agent_components.actions.HRL.custom_actions import CustomActions
from FYP.agent_components.actions.continuous.continuous_actions import ContinuousActions


class ConfigEnv:
    """
    Facilitates setup and configuration of a Gymnasium highway simulation environment. It abstracts configuration
    details such as observation and action spaces, simulation dynamics, and reward settings. This class simplifies
    adjustments and experiments with the environment for training and evaluation purposes.

    Attributes:
        id (str): Identifier for the Gymnasium highway environment.
        config (dict): Environment configuration parameters including simulation settings and reward structures.
    See:
        highway_env.py class for taken config values
    """

    def __init__(self):
        """
        Initializes the ConfigEnv with a predefined set of configuration parameters for the highway environment.
        """
        self.id = 'highway-v0'
        self.config = {
            # do not change:
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
            # ------------------
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
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,  # Terminate episode when offroad
            "screen_width": 1000,
            "screen_height": 500,
            "centering_position": [0.1, 0.5],
            "order": "sorted"
        }

    def create(self, action_type="continuous", render_mode=None, custom_rewards="no"):
        """
        Instantiates and initializes a highway environment with specified configuration and action type, applying a custom
        reward structure. Optionally sets a rendering mode for visual output.

        Args:
            action_type (str, optional): Specifies the type of actions to be used in the environment. `continuous` for
                    default continuous actions, `high-level` for custom high-level actions. Defaults to `continuous`.
            render_mode (str, optional): Render mode (`human`, `rgb_array`, or None) for visual output.
                    Defaults to None, in which case the environment will not render visuals unless explicitly
                    requested later.
            custom_rewards (str, optional): Specifies the type of rewards to be used in the environment. `no` for
                    original rewards from the original gym environment, `yes` for custom rewards. Defaults to `no`.

        Returns:
            gym.Env: Configured Gymnasium environment wrapped with a possibly custom reward and action wrapper,
                    ready for simulation or training.

        """
        env = gym.make(self.id, render_mode=render_mode)
        env.unwrapped.configure(self.config)
        env.reset()

        if custom_rewards == "yes":
            env = CustomReward(env)
        if action_type == "high-level":
            env = CustomActions(env)
        else:
            env = ContinuousActions(env)
        return env
