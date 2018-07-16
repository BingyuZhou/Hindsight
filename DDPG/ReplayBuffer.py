import numpy as np


class ReplayBuffer():
    def __init__(self, replay_buffer_size, n):
        self.replay_buffer_size = replay_buffer_size
        self.end_index = 0
        self.n = n
        self.buffer = np.array([]).reshape((-1, n * 3 + 2))

    def add(self, s, g, a, s_next, r):
        sequence = np.concatenate(
            (s.reshape(1, self.n), g.reshape(1, self.n), a.reshape(1, -1),
             s_next.reshape(1, self.n), np.array([[r]])),
            axis=1)
        # print(sequence)
        if (self.buffer.shape[0] > self.replay_buffer_size):
            self.buffer[int(self.end_index), :] = sequence
            self.end_index = np.mod(self.end_index + 1,
                                    self.replay_buffer_size)
        else:
            self.buffer = np.append(self.buffer, sequence, axis=0)

    def sample(self, size):
        if (self.buffer.shape[0] > size):
            sample_index = np.random.choice(
                self.buffer.shape[0], size, replace=False)
        else:
            sample_index = np.random.choice(
                self.buffer.shape[0], size, replace=True)

        result = {}
        result['s0'] = np.concatenate(
            (self.buffer[sample_index, 0:self.n],
             self.buffer[sample_index, self.n:2 * self.n]),
            axis=1)
        result['s1'] = np.concatenate(
            (self.buffer[sample_index, 2 * self.n + 1:3 * self.n + 1],
             self.buffer[sample_index, self.n:2 * self.n]),
            axis=1)
        result['rewards'] = self.buffer[sample_index, 3 * self.n + 1:]
        result['terminal'] = np.zeros((size, 1))
        result['terminal'][result['rewards'] == 0] = 1
        result['a'] = self.buffer[sample_index, 2 * self.n:2 * self.n + 1]
        return result
