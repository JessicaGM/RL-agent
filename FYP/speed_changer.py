
class SpeedChanger:
    """Class representing behavior for changing speed."""

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