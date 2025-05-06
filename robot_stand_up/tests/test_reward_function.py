import numpy as np
import matplotlib.pyplot as plt
from robot_stand_up.mbrl_main import reward_l1

def test_reward():
    """Test reward when state is exactly at goal height."""
    B = 100
    state = np.zeros((B, 10)) # B x 10
    state[:, 0] = np.linspace(0.0, 0.4, B)
    goal = np.zeros((B, 1))
    goal[:, 0] = 0.2
    reward_values = reward_l1(state, goal.squeeze())

    # Plot the reward values.
    plt.plot(state[:, 0], reward_values, label="L1")
    plt.xlabel("State")
    plt.ylabel("Reward")
    plt.title("Reward vs State")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_reward()