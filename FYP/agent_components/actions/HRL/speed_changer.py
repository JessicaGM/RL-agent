
class SpeedChanger:
    """
    A behavior model for adjusting the speed of a vehicle within a highway environment.

    This class manages the acceleration or deceleration of a vehicle to reach a desired speed,
    taking into account the environment's policy frequency to determine the granularity of speed adjustments.

    Attributes:
        env (Env): The wrapped environment in which the vehicle operates in.
        change (int): Specifies the direction and magnitude of the speed change.
        desired_speed (float): The target speed the vehicle aims to reach.
        step_count (int): Number of steps taken since the beginning of the speed adjustment.
        speed_offset (float): A small margin around the desired speed to account for the granularity of speed adjustments.
    """

    def __init__(self, env, change):
        """
        Initializes a SpeedChanger instance.

        Args:
            env (Env): The wrapped environment in which the vehicle operates in.
            change (int): Specifies the magnitude of the speed change.
        """
        self.env = env
        self.change = change
        self.desired_speed = self.get_current_speed() + change
        self.speed_offset = (1/(self.env.unwrapped.config['policy_frequency']))
        self.step_count = 0

    def get_current_speed(self):
        """Returns the current speed of the vehicle."""
        return self.env.unwrapped.vehicle.speed

    def step(self):
        """
        Executes a step in the environment towards adjusting the vehicle's speed.

        Returns:
            The result of the environment step (observation, reward, done, info).
        """
        self.step_count += 1
        return self.env.step(self.choose_action())

    def done(self):
        """
        Checks if the target speed has been reached within a specified margin.

        Returns:
            bool: True if the speed adjustment is completed, False otherwise.
        """
        done = (self.desired_speed - self.speed_offset <= self.get_current_speed() <= self.desired_speed)

        if done:
            # Reset acceleration to 0 once the target speed is reached
            self.env.unwrapped.vehicle.action['acceleration'] = 0
        return done

    def choose_action(self):
        """
        Determines the low-level action required to adjust the speed towards the desired target.

        Returns:
            list: The action [acceleration, steering] to be taken.
        """
        current_speed = self.get_current_speed()

        steering = self.env.unwrapped.vehicle.action['steering']
        acceleration = self.env.unwrapped.vehicle.action['acceleration']

        # Decide on acceleration based on the current vs. desired speed
        if current_speed < self.desired_speed:
            acceleration = 0.1  # Accelerate
        elif current_speed > self.desired_speed:
            acceleration = -0.1  # Decelerate
        else:
            acceleration = 0  # Maintain current speed

        return [acceleration, steering]