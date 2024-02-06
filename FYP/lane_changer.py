import numpy as np


class LaneChanger:
    """Class representing behavior for changing lanes."""

    def __init__(self, env, change):
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

    def get_current_lane(self):
        """Get the current lane of the vehicle."""
        current_lane = self.env.vehicle.lane_index[2]
        return current_lane

    def step(self):
        """Take a step in the environment."""
        self.step_count += 1
        # print("Step count:", self.step_count)

        return self.env.step(self.choose_action(self.get_current_lane(), self.destination_lane))

    def done(self):
        """Check if the lane change is completed."""
        done = self.destination_lane == self.get_current_lane()
        if done:
            self.env.vehicle.heading = 0  # straighten
            self.env.vehicle.action['steering'] = 0.0
            # print(f"Done: {done}, Current lane: {self.get_current_lane()}, Destination lane: {
            # self.destination_lane}, " f"Vehicle pos: {self.env.vehicle.position}, Steering: {
            # self.env.vehicle.action['steering']}")
        return done

    def choose_action(self, current_lane, destination_lane):
        """Choose the low-level action for the lane change."""
        steering = self.env.vehicle.action['steering']
        acceleration = self.env.vehicle.action['acceleration']

        lane_difference = destination_lane - current_lane
        if lane_difference > 0 or lane_difference < 0:
            steering = np.sign(destination_lane - current_lane) * 0.5 / self.env.vehicle.speed
        elif lane_difference == 0:
            steering = 0

        # print(f"Lanes to destination:
        # {destination_lane - current_lane}, Acceleration: {acceleration}, Steering: {steering}")

        return [acceleration, steering]
