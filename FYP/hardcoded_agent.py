import time
import gymnasium as gym

from FYP.config_env import ConfigEnv
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.behavior import IDMVehicle


class HardcodedAgent:
    """
    A baseline agent that navigates a vehicle through a highway environment using hardcoded logic.

    The agent determines actions (e.g., lane changes, acceleration) based on the current state of the environment,
    specifically focusing on the position and velocity of surrounding vehicles. This agent is useful for establishing
    a performance of the actions.
    Note: The accuracy could be improved by adding more scenarios and quicker actions.

    Attributes:
        env (gym.Env): The environment the agent operates in, wrapped by a custom wrapper to modify its behavior.
        RIGHTMOST_LANE (int): The index of the rightmost lane in the environment.
        LEFTMOST_LANE (int): The index of the leftmost lane, typically 0.
        HIGHEST_AWARDED_SPEED (float): The target speed the agent tries to maintain or reach.
        TIME_TO_CHANGE_LANE (int): The time threshold considered safe for lane changes.
        CONTROLLED_VEHICLE_INDEX (int): The index of the vehicle controlled by the agent within the environment.
    """

    def __init__(self):
        """
        Initializes the agent by setting up the environment and defining key operational parameters.
        """
        self.env = ConfigEnv().make_configured_env(action_type="high-level", render_mode="human")

        # Define operational parameters based on environment configuration
        self.RIGHTMOST_LANE = self.env.unwrapped.config['lanes_count'] - 1
        self.LEFTMOST_LANE = 0
        self.HIGHEST_AWARDED_SPEED = self.env.unwrapped.config['reward_speed_range'][1]
        self.TIME_TO_CHANGE_LANE = 2
        self.CONTROLLED_VEHICLE_INDEX = 0

    def run(self):
        """
        Run the agent in the highway environment.

        The agent takes actions based on simple hardcoded logic, considering the current environment conditions (observation).
        """
        observation = self.env.reset()
        done = False

        while not done:
            # Get information about the controlled vehicle
            vehicle_data = observation[0]

            action = self._select_action(vehicle_data)

            # Take a step in the environment
            obs, reward, done, truncated, info = self.env.step(action)
            observation = obs, info

            # Render the environment
            self.env.render()

        time.sleep(2)
        self.env.close()

    def _select_action(self, vehicle_data):
        """
        Select an action based on the current environment conditions.
        Very simple scenarios.

        Args:
            vehicle_data (array): Information about vehicles in the environment.

        Returns:
            int: The selected action.
        """
        action = 0

        # Check if vehicles in front
        vehicle_in_front = any(
                vehicle[1] + vehicle[3] * self.TIME_TO_CHANGE_LANE <= IDMVehicle.DISTANCE_WANTED
                and AbstractLane.DEFAULT_WIDTH / 2 >= vehicle[2] >= -AbstractLane.DEFAULT_WIDTH / 2
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Check if vehicles on left
        vehicle_on_left = any(
                - AbstractLane.DEFAULT_WIDTH - AbstractLane.DEFAULT_WIDTH / 2 < vehicle[2] < - AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.TIME_TO_CHANGE_LANE <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Check if vehicles on right
        vehicle_on_right = any(
                AbstractLane.DEFAULT_WIDTH + AbstractLane.DEFAULT_WIDTH / 2 > vehicle[2] > AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.TIME_TO_CHANGE_LANE <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Logic for selecting actions based on the assessment of surroundings
        if self.env.vehicle.lane_index[2] < self.RIGHTMOST_LANE and not vehicle_on_right:
            action = 2  # Change to the right lane
        elif (not self.HIGHEST_AWARDED_SPEED - 5 <= self.env.vehicle.speed
              and not self.env.vehicle.speed > self.HIGHEST_AWARDED_SPEED and not vehicle_in_front):
            action = 4  # Accelerate
        elif vehicle_in_front:
            if not vehicle_on_right and self.env.vehicle.lane_index[2] != self.RIGHTMOST_LANE:
                action = 2   # Move to the right lane if safe
            elif not vehicle_on_left and self.env.vehicle.lane_index[2] != self.LEFTMOST_LANE:
                action = 1  # Move to the left lane if safe
            else:
                action = 3  # # Otherwise, slow down (Could have improvements)
        else:
            action = 0  # Go forward

        return action


if __name__ == "__main__":
    agent = HardcodedAgent()
    agent.run()
