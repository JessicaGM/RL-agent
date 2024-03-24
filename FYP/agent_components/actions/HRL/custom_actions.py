import gymnasium as gym
from gymnasium import spaces

from FYP.agent_components.actions.HRL.lane_changer import LaneChanger
from FYP.agent_components.actions.HRL.speed_changer import SpeedChanger
from highway_env.vehicle.kinematics import Vehicle


class CustomActions(gym.ActionWrapper):
    """
    CustomActions wraps the original Gym environment to replace its action space with high-level goals.
    It delegates these high-level goals to specific subpolicy, which encapsulate the logic required
    to perform actions toward achieving these goals.

    Attributes:
        HL_step_count (int): Counter for the number of high-level (HL) steps taken in the current episode.
        LL_step_count (int): Counter for the number of low-level (LL) steps taken in the current episode.
        timesteps_HL (int): Total counter for HL steps taken across all episodes, useful for training.
        timesteps_LL (int): Total counter for LL steps taken across all episodes, useful for training.
        episode_count (int): Counter for the number of episodes processed, useful for training info.
        subpolicies (dict): A mapping from discrete action IDs to their corresponding sub-policy methods.
        action_space (gym.spaces): The action space of the wrapped environment, overridden by the custom action space.
        leftmost_lane (int): Index of the leftmost lane in the environment.
        rightmost_lane (int): Index of the rightmost lane in the environment.

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
        self.timesteps_HL = 0
        self.timesteps_LL = 0
        self.episode_count = 0
        self.subpolicies = self.HL_actions()
        self.action_space = spaces.Discrete(len(self.subpolicies))
        self.leftmost_lane = 0
        self.rightmost_lane = self.env.unwrapped.config['lanes_count'] - 1

    def reset(self, **kwargs):
        """
        Resets the environment and step counters.

        Args:
            **kwargs: Additional arguments to the environment's reset method.

        Returns:
             tuple: A tuple of the initial observation from the environment and dictionary of info.
        """
        self.HL_step_count = 0
        self.LL_step_count = 0
        self.episode_count += 1

        return self.env.reset(**kwargs)

    def HL_actions(self):
        """
        High-level action/goal.

        Returns:
            dict: A dictionary mapping action IDs to their corresponding methods that contain the subpolicy.
        """
        subpolicies = {
            0: self.forward_10_meters,
            1: self.change_to_left_lane,
            2: self.change_to_right_lane,
            3: self.slow_down,
            4: self.speed_up
        }
        return subpolicies

    def forward_10_meters(self):
        """
        Perform a lane change action to move forward by approximately 10 meters.

        Returns:
            LaneChanger: Instance of LaneChanger - subpolicy for moving forward.
        """
        return LaneChanger(self.env, 0, 10)

    def change_to_left_lane(self):
        """
        Perform a lane change action to move to the left lane.

        Returns:
            LaneChanger: Instance of LaneChanger - subpolicy for changing to the left lane.
        """
        return LaneChanger(self.env, -1)

    def change_to_right_lane(self):
        """
        Perform a lane change action to move to the right lane.

        Returns:
            LaneChanger: Instance of LaneChanger - subpolicy for changing to the right lane.
        """
        return LaneChanger(self.env, 1)

    def slow_down(self):
        """
        Perform a speed change action to slow down by approximately 1 m/s.

        Returns:
            SpeedChanger: Instance of SpeedChanger - subpolicy for slowing down.
        """
        return SpeedChanger(self.env, -1)

    def speed_up(self):
        """
        Perform a speed change action to speed up by approximately 1 m/s.

        Returns:
            SpeedChanger: Instance of SpeedChanger - subpolicy for speeding up.
        """
        return SpeedChanger(self.env, 1)

    def lane_change_possible(self, change):
        """
        Validates if a lane change action is within the permissible range.

        Returns:
            bool: True if the lane change is possible, False otherwise.
        """
        destination_lane = self.env.unwrapped.vehicle.lane_index[2] + change
        return self.leftmost_lane <= destination_lane <= self.rightmost_lane

    def speed_change_possible(self, change):
        """
        Validates if a speed change action is within the permissible range.

        Returns:
            bool: True if the speed change is possible, False otherwise.
        """
        destination_speed = self.env.unwrapped.vehicle.speed + change
        return Vehicle.MIN_SPEED <= destination_speed <= Vehicle.MAX_SPEED

    def step(self, action):
        """
        Performs a high-level action in the environment using appropriate subpolicy and updates step counts.

        Returns: A tuple of:
            obs: The observation after taking the action.
            cumulative_reward: The reward obtained after executing the high-level action (all low-level actions).
            terminated: A boolean indicating if the episode has ended.
            truncated: A boolean indicating if the episode was truncated.
            info: A dictionary with additional information about the step.
        """
        changer = self.subpolicies[int(action)]()

        # Check if action is possible before execution, adjust if necessary
        if isinstance(changer, LaneChanger) and not self.lane_change_possible(changer.change):
            changer = LaneChanger(self.env, 0)
        if isinstance(changer, SpeedChanger) and not self.speed_change_possible(changer.change):
            changer = SpeedChanger(self.env, 0)

        # Execute low-level until high-level action is done
        obs, reward, terminated, truncated, info = changer.step()
        cumulative_reward = reward
        while not changer.done() and not terminated:
            obs, reward, terminated, truncated, info = changer.step()
            cumulative_reward += reward

        self.LL_step_count += changer.step_count
        self.timesteps_LL += changer.step_count
        self.HL_step_count += 1
        self.timesteps_HL += 1

        # Useful for evaluation logs
        info["HL_step_count"] = self.HL_step_count
        info["LL_step_count"] = self.LL_step_count
        info["pos_x"] = obs[0][1]
        info["pos_y"] = obs[0][2]

        return obs, cumulative_reward, terminated, truncated, info
