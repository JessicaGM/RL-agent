from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    Custom callback for logging additional information during training.

    Attributes:
        env_wrapper: The wrapped environment to access additional data.
        action_type: The action type.
    """

    def __init__(self, env_wrapper, action_type, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env_wrapper = env_wrapper
        self.action_type = action_type

    def _on_step(self) -> bool:
        """
        Logs the info to TensorBoard.

        Returns:
            bool: True to continue training, False otherwise.
        """
        if self.env_wrapper is not None:
            if self.action_type == "high-level":
                self.logger.record("custom/HL_step_count", self.env_wrapper.HL_step_count)
                self.logger.record("custom/LL_step_count", self.env_wrapper.LL_step_count)
                self.logger.record("custom/episode_count", self.env_wrapper.episode_count)
                self.logger.record("custom/timesteps_HL", self.env_wrapper.timesteps_HL)
                self.logger.record("custom/timesteps_LL", self.env_wrapper.timesteps_LL)
            if self.action_type == "continuous":
                self.logger.record("custom/step_count", self.env_wrapper.step_count)
                self.logger.record("custom/episode_count", self.env_wrapper.episode_count)
                self.logger.record("custom/timesteps", self.env_wrapper.timesteps)
        return True
