import time
import gymnasium as gym
import numpy as np

from highway_env.envs import HighwayEnv
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.behavior import IDMVehicle

highway_env_instance = HighwayEnv()


class LaneChanger:
    """Class representing a behavior for changing lanes."""

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


class SpeedChanger:
    """Class representing a behavior for changing speed."""

    def __init__(self, env, change):
        """
        Initialize the SpeedChanger.

        Args:
            env (HighwayEnv): The environment.
            change (int): The speed change direction (-1 for deceleration, 1 for acceleration).
        """
        self.env = env
        self.desired_speed = self.get_current_speed() + change
        self.step_count = 0

    def get_current_speed(self):
        """Get the current speed of the vehicle."""
        current_speed = self.env.vehicle.speed
        return current_speed

    def step(self):
        """Take a step in the environment."""
        self.step_count += 1
        # print("Step count:", self.step_count)
        return self.env.step(self.choose_action(self.get_current_speed(), self.desired_speed))

    def done(self):
        """Check if the desired speed is reached."""
        return self.desired_speed - 0.5 < self.get_current_speed() <= self.desired_speed

    def choose_action(self, current_speed, desired_speed):
        """Choose the low-level action for changing speed."""
        steering = self.env.vehicle.action['steering']
        acceleration = self.env.vehicle.action['acceleration']

        # print(f"Initial - steering: {steering} & acceleration: {acceleration}")

        if current_speed < desired_speed:
            acceleration = 0.1
        elif current_speed > desired_speed:
            acceleration = -0.1
        elif current_speed == desired_speed:
            acceleration = 0

        # print(f"Acceleration: {acceleration}, Steering: {steering}, "
        #     f"Current speed: {current_speed}, Desired speed: {desired_speed}")
        return [acceleration, steering]


class CustomWrapper(gym.Wrapper):
    """
    A custom wrapper for modifying the behavior of a highway environment.

    This wrapper extends the gym.Wrapper class and is designed to modify the behavior of the underlying
    highway environment.

    Attributes:
        env (HighwayEnv): The original highway environment.

    Methods:
        reset(**kwargs):
            Reset the environment and return the initial observation.

        step(action):
            Take a step in the environment based on the provided action.

    Usage:
        env = gym.make("highway-v0", render_mode="human")
        env = CustomWrapper(env)
    """

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset the environment."""
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Take a step in the environment based on the provided action."""
        self.step_count += 1
        # print("Step count:", self.step_count)

        terminated = False
        change = 0

        # Forward or Change lane
        if 0 <= action <= 2:
            # Go forward
            if action == 0:
                change = 0
            # Change left lane
            if action == 1:
                change = -1
            # Change right lane
            if action == 2:
                change = 1
            changer = LaneChanger(self.env, change)
            obs, reward, terminated, truncated, info = changer.step()
            # print(obs, info)
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
                # print(obs, info)
            return obs, reward, terminated, truncated, info

        # Adjust speed
        if 3 <= action <= 6:
            # Decelerate
            if action == 3:
                change = -1  # change speed by -1
            # Accelerate
            if action == 4:
                change = 1  # change speed by 1
            # Maintain speed
            if action == 5:
                change = 0  # no change
            # Decelerate quicker
            changer = SpeedChanger(self.env, change)
            # print(obs, info)
            obs, reward, terminated, truncated, info = changer.step()
            while not changer.done() and not terminated:
                obs, reward, terminated, truncated, info = changer.step()
                # print(obs, info)
            return obs, reward, terminated, truncated, info


env = gym.make("highway-v0", render_mode="human")
env = CustomWrapper(env)

observation = env.reset()

done = False
while not done:

    # print(observation)

    # Simple hardcoded action selection

    RIGHT_MOST_LANE = (env.unwrapped.config['lanes_count'] - 1)
    LEFT_MOST_LANE = 0
    HIGHEST_AWARDED_SPEED = (env.unwrapped.config['reward_speed_range'][1])

    action = 0

    vehicle_data = observation[0]
    controlled_vehicle_index = 0
    controlled_vehicle = vehicle_data[controlled_vehicle_index]
    presence, x, y, vx, vy = controlled_vehicle

    time_to_change_lane = 2

    vehicle_in_front = any(vehicle[1] + vehicle[3] * time_to_change_lane <= IDMVehicle.DISTANCE_WANTED
                           and AbstractLane.DEFAULT_WIDTH / 2 >= vehicle[2] >= -AbstractLane.DEFAULT_WIDTH / 2
                           and vehicle[0] == 1
                           for i, vehicle in enumerate(vehicle_data) if i != controlled_vehicle_index)
    # print(f"Is there a vehicle in front of the car in distance: {IDMVehicle.DISTANCE_WANTED}?: {vehicle_in_front}")

    # Check if vehicles on left
    vehicle_on_left = any(
        - AbstractLane.DEFAULT_WIDTH - AbstractLane.DEFAULT_WIDTH / 2 < vehicle[2] < - AbstractLane.DEFAULT_WIDTH / 2
        and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * time_to_change_lane <= IDMVehicle.DISTANCE_WANTED
        and vehicle[0] == 1
        for i, vehicle in enumerate(vehicle_data) if i != controlled_vehicle_index)
    # print(f"Is there a vehicle on the left?: {vehicle_on_left}")

    # Check if vehicles on right
    vehicle_on_right = any(
        AbstractLane.DEFAULT_WIDTH + AbstractLane.DEFAULT_WIDTH / 2 > vehicle[2] > AbstractLane.DEFAULT_WIDTH / 2
        and -IDMVehicle.DISTANCE_WANTED <= vehicle[1] + vehicle[3] * time_to_change_lane <= IDMVehicle.DISTANCE_WANTED
        and vehicle[0] == 1
        for i, vehicle in enumerate(vehicle_data) if i != controlled_vehicle_index)
    # print(f"Is there a vehicle on the right?: {vehicle_on_right}")

    # If vehicle not at rightmost lane then change to right lane
    if env.vehicle.lane_index[2] < RIGHT_MOST_LANE and not vehicle_on_right:
        action = 2
        # print("Change to right lane")
    # If vehicle not between 25-30 and not higher than 30, accelerate
    elif not HIGHEST_AWARDED_SPEED - 5 <= env.vehicle.speed and not env.vehicle.speed > HIGHEST_AWARDED_SPEED and not vehicle_in_front:
        action = 4
        # print("Accelerate")
    # If vehicle in front
    elif vehicle_in_front:
        # print("Vehicle in front")
        if not vehicle_on_right and env.vehicle.lane_index[2] != RIGHT_MOST_LANE:
            action = 2
            # print("Right lane empty, go to right lane")
        elif not vehicle_on_left and env.vehicle.lane_index[2] != LEFT_MOST_LANE:
            action = 1
            # print("Left lane empty, go to left lane")
        else:
            action = 6
            # print("Slow down")
    else:
        action = 0

    obs, reward, done, truncated, info = env.step(action)

    observation = obs, info

    env.render()

time.sleep(2)
env.close()
