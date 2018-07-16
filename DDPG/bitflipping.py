"""
Bit-flipping Environment

- Given a n bit number and a goal number, find a sequence of actions to transfer the inital number to the goal number.
- State space:
    S = {0,1}^n
- Action space:
    A = {0,1,...,n-1}
    where executing the i-th action flips the i-th bit of state
- Reward:
    -1 if the current number is not the goal number
"""

import numpy as np
import random


class bitflipping:
    def __init__(self, length):
        self.n = length
        state = self._sample_state()
        goal = self._sample_state()

        while np.array_equal(state, goal):
            goal = self._sample_state()

        self.state = np.copy(state)
        self.goal = np.copy(goal)

    def _sample_state(self):
        """
        Sample a sequence of bits with certain length
        """
        state = []
        for i in range(self.n):
            if (random.random() < 0.5):
                state.append(1)
            else:
                state.append(0)

        return np.asarray(state)

    def update_state(self, action):
        assert action < self.n, "Action is not allowed!"
        self.state[int(
            np.around(action))] = 1 - self.state[int(np.around(action))]
        return np.copy(self.state)

    def reward(self, state, goal=None):
        if goal is None:
            if np.array_equal(state, self.goal):
                return 0
            else:
                return -1
        else:
            if np.array_equal(state, goal):
                return 0
            else:
                return -1

    def reset(self):
        state = self._sample_state()
        goal = self._sample_state()
        while np.array_equal(state, goal):
            goal = self._sample_state()
        self.state = np.copy(state)
        self.goal = np.copy(goal)
