import time

import numpy as np

from highway_env.road.lane import AbstractLane


class LaneChanger:
    """Class representing behavior for changing lanes."""

    def __init__(self, env, change, distance):
        """
        Initialize the LaneChanger.

        Args:
            env (HighwayEnv): The environment.
            change (int): The lane change direction (-1 for left, 1 for right).
        """
        self.env = env
        self.change = change
        self.destination_lane = self.get_current_lane() + self.change
        self.step_count = 0
        self.vehicle_posX = self.get_current_posX()
        self.target_distance = distance

    def get_current_lane(self):
        """Get the current lane of the vehicle."""
        current_lane = self.env.unwrapped.vehicle.lane_index[2]
        return current_lane

    def get_current_posX(self):
        """Get the current position x of the vehicle."""
        current_posX = self.env.unwrapped.vehicle.position[0]
        return current_posX

    def step(self):
        """Take a step in the environment."""
        self.step_count += 1
        # print("Step count:", self.step_count)
        return self.env.step(self.choose_action(self.get_current_lane(), self.destination_lane))

    def done(self):
        """Check if the lane change is completed."""
        done = (self.destination_lane == self.get_current_lane() and 0.5 > self.env.unwrapped.vehicle.lane_offset[1] > -0.5)
        if done:
            if self.change == 0:
                distance_covered = self.get_current_posX() - self.vehicle_posX

                if distance_covered >= self.target_distance:
                    self._reset_after_change()
                    # print(f"Done: Forward travel completed. Distance covered: {distance_covered}")
                    return True
                else:
                    # print(f"Continue forward: Distance covered: {distance_covered}")
                    return False
            else:
                self._reset_after_change()
                # print("Done: Desired lane reached.")

        return done

    def _reset_after_change(self):
        """Reset after completing lane change."""
        self.env.unwrapped.vehicle.heading = 0  # straighten
        self.env.unwrapped.vehicle.action['steering'] = 0.0
        # print(f"Done: {done}, Current lane: {self.get_current_lane()},
        # Destination lane: {self.destination_lane}, Vehicle pos: {self.env.vehicle.position},
        # Steering: {self.env.vehicle.action['steering']}")

    def choose_action(self, current_lane, destination_lane):
        """Choose the low-level action for the lane change."""
        steering = self.env.unwrapped.vehicle.action['steering']
        acceleration = self.env.unwrapped.vehicle.action['acceleration']

        lane_difference = destination_lane - current_lane
        if lane_difference > 0 or lane_difference < 0:
            steering = np.sign(destination_lane - current_lane) * 0.5 / self.env.unwrapped.vehicle.speed
        elif lane_difference == 0:
            steering = 0

        # print(f"Lanes to destination: {destination_lane - current_lane}, Acceleration: {acceleration}, Steering: {steering}")

        return [acceleration, steering]
