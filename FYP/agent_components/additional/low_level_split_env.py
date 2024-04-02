import gymnasium as gym


class LowLevelSplitEnv(gym.ActionWrapper):
    """
    Additional functionality not yet finished.
    """

    def __init__(self, env, change):
        """
        Initializes the custom wrapper.

        Args:
            env (gym.Env): The original Gymnasium environment to be wrapped.
        """
        super().__init__(env)
        self.step_count = 0
        self.timesteps = 0
        self.episode_count = 0
        self.done_count_all_episodes = 0
        self.change = change
        self.desired_speed = self.get_current_speed() + self.change
        self.speed_offset = (1 / (self.env.unwrapped.config['policy_frequency']))
        self.last_speed = self.get_current_speed()
        self.initial_vehicle_lane = self.env.unwrapped.vehicle.lane_index

    def reset(self, **kwargs):
        """
        Resets the environment and step counters.

        Returns:
           The initial env.
        """
        self.step_count = 0
        self.episode_count += 1
        self.initial_vehicle_lane = self.env.unwrapped.vehicle.lane_index
        self.desired_speed = self.get_current_speed() + self.change
        self.last_speed = self.get_current_speed()
        return self.env.reset(**kwargs)

    def get_current_speed(self):
        """Returns the current speed of the vehicle."""
        return self.env.unwrapped.vehicle.speed

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
            print("Done")
            self.done_count_all_episodes += 1
        return done

    def calculate_reward(self, action):
        """
        Calculates the reward based on the change in speed and lane maintaining.

        Returns:
            int: The reward value.
        """
        current_speed = self.get_current_speed()

        speed_reward_value = 1
        other_reward_value = 1

        speed_reward = 0
        other_reward = 0

        if ((self.change < 0 and (self.desired_speed or self.desired_speed - self.speed_offset)  # for action slow down
             <= current_speed < self.last_speed)
                or (self.change > 0 and (
                        self.desired_speed or self.desired_speed - self.speed_offset)  # for action speed up
                    <= current_speed > self.last_speed)):
            speed_reward = speed_reward_value
        self.last_speed = current_speed

        if self.env.unwrapped.vehicle.lane_index == self.initial_vehicle_lane:
            other_reward = 1

        reward = speed_reward * other_reward

        print(f"E: {self.episode_count}, Action: {action} "
              f"Reward: {reward}, "
              f"S: {speed_reward}, "
              f"L: {other_reward}, "
              f"Desired speed: {self.desired_speed}, "
              f"Current speed: {self.get_current_speed()}, Heading: {self.env.unwrapped.vehicle.heading}, "
              f"Initial lane: {self.initial_vehicle_lane}, Lane: {self.env.unwrapped.vehicle.lane_index}")

        return reward

    def step(self, action):
        """
        Takes an action in the environment and updates counters based on the result.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the observation, reward, terminated, truncated, and info flags.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        self.timesteps += 1

        reward = self.calculate_reward(action)

        # Check if speed condition is met
        if self.done():
            terminated = True

        return obs, reward, terminated, truncated, info
