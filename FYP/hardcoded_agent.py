import time
import gymnasium as gym

from FYP.config_env import ConfigEnv
from custom_wrapper import CustomWrapper
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.behavior import IDMVehicle


class HardcodedAgent:
    """
    An agent that controls a vehicle in a highway environment based on hardcoded action selection.

    Attributes:
        env (CustomWrapper): The custom-wrapped highway environment.
        RIGHT_MOST_LANE (int): The index of the rightmost lane.
        LEFT_MOST_LANE (int): The index of the leftmost lane.
        HIGHEST_AWARDED_SPEED (float): The highest speed for rewarding the agent.
        TIME_TO_CHANGE_LANE (int): Time considered to change lane.
        CONTROLLED_VEHICLE_INDEX (int): Index of the controlled vehicle.

    Methods:
        run():
            Run the agent in the highway environment.
    """

    def __init__(self):
        """
        Initialize the Agent.

        This method sets up the highway environment, custom wrapper, and defines parameters.
        """
        self.env = ConfigEnv().make_configured_env(render_mode="human")
        self.env = CustomWrapper(self.env)

        # Environment parameters
        self.RIGHT_MOST_LANE = self.env.unwrapped.config['lanes_count'] - 1
        self.LEFT_MOST_LANE = 0
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
        # print(f"Is there a vehicle in front of the car in distance: {IDMVehicle.DISTANCE_WANTED}?: {vehicle_in_front}")

        # Check if vehicles on left
        vehicle_on_left = any(
                - AbstractLane.DEFAULT_WIDTH - AbstractLane.DEFAULT_WIDTH / 2 < vehicle[2] < - AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.TIME_TO_CHANGE_LANE <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))
        # print(f"Is there a vehicle on the left?: {vehicle_on_left}")

        # Check if vehicles on right
        vehicle_on_right = any(
                AbstractLane.DEFAULT_WIDTH + AbstractLane.DEFAULT_WIDTH / 2 > vehicle[2] > AbstractLane.DEFAULT_WIDTH / 2
                and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * self.TIME_TO_CHANGE_LANE <= IDMVehicle.DISTANCE_WANTED
                and vehicle[0] == 1
                for i, vehicle in enumerate(vehicle_data[1:]))
        # print(f"Is there a vehicle on the right?: {vehicle_on_right}")

        # If vehicle not at rightmost lane then change to right lane
        if self.env.vehicle.lane_index[2] < self.RIGHT_MOST_LANE and not vehicle_on_right:
            action = 2
            # print("Change to right lane")

        # If vehicle not between 25-30 and not higher than 30, accelerate
        elif (not self.HIGHEST_AWARDED_SPEED - 5 <= self.env.vehicle.speed
              and not self.env.vehicle.speed > self.HIGHEST_AWARDED_SPEED and not vehicle_in_front):
            action = 4
            # print("Speed up")

        elif vehicle_in_front:
            # print("Vehicle in front")
            if not vehicle_on_right and self.env.vehicle.lane_index[2] != self.RIGHT_MOST_LANE:
                action = 2
                # print("Right lane empty, go to right lane")
            elif not vehicle_on_left and self.env.vehicle.lane_index[2] != self.LEFT_MOST_LANE:
                action = 1
                # print("Left lane empty, go to left lane")
            else:
                action = 3  # Could have improvements
                # print("Slow down")
        else:
            action = 0

        return action


if __name__ == "__main__":
    agent = HardcodedAgent()
    agent.run()
