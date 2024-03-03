import time

import numpy as np

from highway_env.road.lane import AbstractLane


class LaneChanger:
    """
    A behavior model for changing lanes or moving forward a specified distance within a highway environment.

    Attributes:
        env (Environment): The simulation environment containing the vehicle.
        change (int): Direction and magnitude of lane change (-1 for left, 1 for right, 0 for no lane change).
        destination_lane (int): The target lane after the change.
        step_count (int): Number of steps taken since the beginning of the action.
        vehicle_posX (float): The initial x-position of the vehicle when the action started.
        target_distance (float): The target distance to move forward if no lane change is specified.
    """

    def __init__(self, env, change, distance=None):
        """
        Initializes a LaneChanger instance.

        Args:
            env: The environment in which the vehicle operates.
            change (int): Specifies the direction and magnitude of the lane change (-1, 0, 1).
            distance (float, optional): The distance to move forward if no lane change is required.
        """
        self.env = env
        self.change = change
        self.destination_lane = self.get_current_lane() + self.change
        self.step_count = 0
        self.vehicle_posX = self.get_current_posX()
        self.target_distance = distance or 0  # Default to 0 if no distance is provided

    def get_current_lane(self):
        """Returns the current lane index of the vehicle."""
        return self.env.unwrapped.vehicle.lane_index[2]

    def get_current_posX(self):
        """Returns the current x-position of the vehicle."""
        return self.env.unwrapped.vehicle.position[0]

    def step(self):
        """
        Executes a step in the environment towards completing the lane change or moving forward.

        Returns:
            The result of the environment step (observation, reward, done, info).
        """
        self.step_count += 1
        return self.env.step(self.choose_action())

    def done(self):
        """
        Checks if the lane change or forward movement is completed.

        Returns:
            bool: True if the action is completed, False otherwise.
        """
        done = (self.destination_lane == self.get_current_lane() and 0.5 > self.env.unwrapped.vehicle.lane_offset[1] > -0.5)
        if done:
            if self.change == 0:
                distance_covered = self.get_current_posX() - self.vehicle_posX

                if distance_covered >= self.target_distance:
                    self._reset_after_change()
                    return True
                else:
                    return False
            else:
                self._reset_after_change()

        return done

    def _reset_after_change(self):
        """Resets the vehicle's heading and steering after completing a lane change."""
        self.env.unwrapped.vehicle.heading = 0  # straighten
        self.env.unwrapped.vehicle.action['steering'] = 0.0

    def choose_action(self):
        """
        Determines the steering action required to change lanes or continue moving forward.
        Low-level action.

        Returns:
            list: The action [acceleration, steering] to be taken.
        """
        steering = self.env.unwrapped.vehicle.action['steering']
        acceleration = self.env.unwrapped.vehicle.action['acceleration']

        current_lane = self.get_current_lane()
        lane_difference = self.destination_lane - current_lane
        if lane_difference > 0 or lane_difference < 0:
            steering = np.sign(self.destination_lane - current_lane) * 0.5 / self.env.unwrapped.vehicle.speed
        elif lane_difference == 0:
            steering = 0     # Straighten up if in the target lane or moving forward

        return [acceleration, steering]
