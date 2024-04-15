import gymnasium as gym
import pytest

from highway_env.vehicle.kinematics import Vehicle
from FYP.agent_components.actions.HRL.custom_actions import CustomActions


@pytest.fixture
def wrapped_env():
    """
    Set up of the environment with initial lane of controlled vehicle set to 1
    and no other vehicles on the road for testing lane changing and speeding.
    """
    config = {
        "initial_lane_id": 0,
        "vehicles_count": 0
    }
    env = gym.make('highway-v0')
    env.configure(config)
    env.reset()
    env = CustomActions(env)
    return env


def test_reset(wrapped_env):
    """Test the reset method."""
    wrapped_env.reset()
    assert wrapped_env.HL_step_count == 0
    assert wrapped_env.LL_step_count == 0


def test_initialise_subpolicies(wrapped_env):
    """Test if subpolicies are initialized correctly."""
    subpolicies = wrapped_env.HL_actions()
    assert isinstance(subpolicies, dict)
    assert len(subpolicies) == 5


def test_lane_change_possible(wrapped_env):
    """Test lane change possible method."""
    # Change to left lane
    assert not wrapped_env.lane_change_possible(wrapped_env.subpolicies[1]().change)
    assert not wrapped_env.lane_change_possible(-1)
    # Change to right lane
    assert wrapped_env.lane_change_possible(wrapped_env.subpolicies[2]().change)
    assert wrapped_env.lane_change_possible(1)


def test_speed_change_possible(wrapped_env):
    """Test speed change possible method."""
    # Min speed reached
    wrapped_env.vehicle.speed = Vehicle.MIN_SPEED
    # Slow down by 1m/s
    assert not wrapped_env.speed_change_possible(wrapped_env.subpolicies[3]().change)
    # Speed up by 1m/s
    assert wrapped_env.speed_change_possible(wrapped_env.subpolicies[4]().change)

    # Max speed reached
    wrapped_env.vehicle.speed = Vehicle.MAX_SPEED
    # Slow down by 1m/s
    assert wrapped_env.speed_change_possible(wrapped_env.subpolicies[3]().change)
    # Speed up by 1m/s
    assert not wrapped_env.speed_change_possible(wrapped_env.subpolicies[4]().change)
