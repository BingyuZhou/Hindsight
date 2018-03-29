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

class bit_flipping:
    def __init__(self, init_state, goal, length):
        self.state = init_state
        self.goal = goal
        self.n = length #length of bits
    
    def update_state(self, action):
        state = self.state
        assert action<self.n, 'action is not allowed!'

        state[action] = 1-state[action]
        self.state = state
        return state
    
    def reward(self, state):
        if state==self.goal:
            return 0
        else:
            return -1

