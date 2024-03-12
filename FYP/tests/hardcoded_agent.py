import time
import gymnasium as gym

from FYP.agent_components.config_env import ConfigEnv
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.behavior import IDMVehicle


class HardcodedAgent:
    """
    A baseline agent that navigates a vehicle through a highway environment using hardcoded logic.
    Mainly used to test hardcoded low-level actions.

    The agent determines actions (e.g., lane changes, acceleration) based on the current state of the environment,
    specifically focusing on the position and velocity of surrounding vehicles. This agent is useful for establishing
    a performance of the actions.
    Note: The accuracy could be improved by adding more scenarios and quicker actions.

    Attributes:
        env (gym.Env): The environment the agent operates in, wrapped by a custom wrapper to modify its behavior.
        right_most_lane (int): The index of the right_most lane in the environment.
        left_most_lane (int): The index of the left_most lane, typically 0.
        highest_awarded_speed (float): The target speed the agent tries to maintain or reach.
        time_to_change (int): The time threshold considered safe for lane changes.
    """

    def __init__(self):
        """
        Initializes the agent by setting up the environment and defining key operational parameters.
        """
        self.env = ConfigEnv().create(action_type="high-level", render_mode="human")

        # Define operational parameters based on environment configuration
        self.right_most_lane = self.env.unwrapped.config['lanes_count'] - 1
        self.left_most_lane = 0
        self.highest_awarded_speed = self.env.unwrapped.config['reward_speed_range'][1]
        self.time_to_change = 2     # e.g.

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

            action = self.select_action(vehicle_data)

            # Take a step in the environment
            obs, reward, done, truncated, info = self.env.step(action)
            observation = obs, info

            # Render the environment
            self.env.render()

        time.sleep(2)
        self.env.close()

    def select_action(self, vehicle_data):
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
                vehicle[1] + vehicle[3] * self.time_to_change <= IDMVehicle.DISTANCE_WANTED
                and AbstractLane.DEFAULT_WIDTH / 2 >= vehicle[2] >= -AbstractLane.DEFAULT_WIDTH / 2
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Check if vehicles on left
        vehicle_on_left = any(
                - AbstractLane.DEFAULT_WIDTH - AbstractLane.DEFAULT_WIDTH / 2 < vehicle[2] < - AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.time_to_change <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Check if vehicles on right
        vehicle_on_right = any(
                AbstractLane.DEFAULT_WIDTH + AbstractLane.DEFAULT_WIDTH / 2 > vehicle[2] > AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.time_to_change <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))

        # Logic for selecting actions based on the assessment of surroundings
        if self.env.unwrapped.vehicle.lane_index[2] < self.right_most_lane and not vehicle_on_right:
            action = 2  # Change to the right lane
        elif (not self.highest_awarded_speed - 5 <= self.env.unwrapped.vehicle.speed
              and not self.env.unwrapped.vehicle.speed > self.highest_awarded_speed and not vehicle_in_front):
            action = 4  # Accelerate
        elif vehicle_in_front:
            if not vehicle_on_right and self.env.unwrapped.vehicle.lane_index[2] != self.right_most_lane:
                action = 2   # Move to the right lane if safe
            elif not vehicle_on_left and self.env.unwrapped.vehicle.lane_index[2] != self.left_most_lane:
                action = 1  # Move to the left lane if safe
            else:
                action = 3  # # Otherwise, slow down (Could have improvements)
        else:
            action = 0  # Go forward

        return action


if __name__ == "__main__":
    agent = HardcodedAgent()
    agent.run()
