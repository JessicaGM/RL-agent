class LaneChanger:
    """
    A behavior model for changing lanes or moving forward a specified distance within a highway environment.

    This class allows a simulated vehicle to perform lane changes to the left or right, or to move straight ahead for
    a given distance, based on the environment's configuration and the vehicle's current state. It takes into account
    the policy frequency of the environment to calculate a lane position tolerance, ensuring smooth transitions.

    Attributes:
        env (Environment): The simulation environment containing the vehicle.
        change (int): Specifies the direction and magnitude of the lane change (-1 for left, 1 for right, 0 for no lane
                    change). This determines the target lane relative to the vehicle's current lane.
        destination_lane (int): The index of the target lane after the change has been made. Calculated based on
                    the current lane and the change direction.
        step_count (int): Tracks the number of steps taken since the beginning of the action. This is used to
                    monitor progress towards the action's completion.
        vehicle_posX (float): Records the initial x-position (longitudinal position) of the vehicle when the lane change
                    or forward movement action started.
        target_distance (float): The target distance to move forward if no lane change is specified. This is used to
                    determine when a forward movement action is complete.
        lane_position_tolerance (float): A calculated tolerance value based on the environment's policy frequency.
                    It's used to determine if the vehicle is sufficiently close to the center of the target lane or
                    has moved the required distance.
    """

    def __init__(self, env, change, distance=None):
        """
        Initializes a LaneChanger instance with a specified environment, lane change direction, and forward
        movement distance.

        Args:
            env: The simulation environment in which the vehicle operates.
            change (int): Magnitude of the lane change.
            distance (float, optional): The distance (in meters) the vehicle should move forward if no lane change
                    is required. Defaults to 0.
        """
        self.env = env
        self.change = change
        self.destination_lane = self.get_current_lane() + self.change
        self.step_count = 0
        self.vehicle_posX = self.get_current_posX()
        self.target_distance = distance or 0
        self.lane_position_tolerance = (1 / (self.env.unwrapped.config['policy_frequency'])) * 1.5

    def get_current_lane(self):
        """
        Retrieves the current lane index of the vehicle within the environment.

        Returns:
            int: The index of the lane in which the vehicle is currently located.
        """
        return self.env.unwrapped.vehicle.lane_index[2]

    def get_current_posX(self):
        """
        Obtains the current longitudinal position of the vehicle in the environment.

        Returns:
            float: The vehicle's current x-position.
        """
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
        Checks whether the specified lane change or forward movement has been completed.

        For lane changes, completion is determined by the vehicle's presence within the target lane and within
        the `lane_position_tolerance`.
        For forward movements, completion is determined by the vehicle having moved the specified `target_distance`
        from its initial position.

        Returns:
            bool: True if the specified action (lane change or forward movement) is completed, False otherwise.
        """
        done = (self.destination_lane == self.get_current_lane() and
                self.lane_position_tolerance > self.env.unwrapped.vehicle.lane_offset[1] > -self.lane_position_tolerance)

        if done and self.change == 0:
            distance_covered = self.get_current_posX() - self.vehicle_posX
            return distance_covered >= self.target_distance

        self.reset_after_change() if done else None
        return done

    def reset_after_change(self):
        """
        Resets the vehicle's heading and steering to default values after completing a lane change.

        This ensures that the vehicle aligns properly with the lane and is ready for subsequent actions.
        """
        self.env.unwrapped.vehicle.heading = 0  # Reset heading to straight forward
        self.env.unwrapped.vehicle.action['steering'] = 0.0  # Reset steering to neutral

    def choose_action(self):
        """
        Determines the low-level action to achieve the desired lane change or forward movement.

        Returns:
           list: The action [acceleration, steering] to be taken.
        """
        steering = self.env.unwrapped.vehicle.action['steering']
        acceleration = self.env.unwrapped.vehicle.action['acceleration']

        lane_difference = self.destination_lane - self.get_current_lane()
        if lane_difference > 0:
            steering = 0.5 / self.env.unwrapped.vehicle.speed  # Adjust steering for right lane change
        elif lane_difference < 0:
            steering = -0.5 / self.env.unwrapped.vehicle.speed  # Adjust steering for left lane change
        else:
            steering = 0  # No steering adjustment needed for straight movement

        return [acceleration, steering]
