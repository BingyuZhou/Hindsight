import numpy as np
import random


def her(buffer, method, n, reward_fun):
    def _her_final(buffer, n, reward_fun):

        if buffer.end_index == 0:
            her_ind = buffer.buffer.shape[0]
        else:
            her_ind = buffer.end_index

        goal_her = np.copy(buffer.buffer[her_ind - 1, 2 * n + 1:3 * n + 1])

        for t in range(n):

            replay_buffer_row = her_ind - t - 1
            if replay_buffer_row < 0:
                replay_buffer_row += buffer.replay_buffer_size

            # print(replay_buffer_row)
            # print(her_ind)

            s = buffer.buffer[int(replay_buffer_row), 0:n]
            a = buffer.buffer[int(replay_buffer_row), 2 * n]
            s_next = buffer.buffer[int(replay_buffer_row), 2 * n + 1:3 * n + 1]

            reward_her = reward_fun(s_next, goal_her)

            buffer.add(s, goal_her, a, s_next, reward_her)
        return buffer

    def _her_future(buffer, k, n, reward_fun):
        if buffer.end_index == 0:
            her_ind = buffer.buffer.shape[0]
        else:
            her_ind = buffer.end_index

        for t in range(n):
            selected_rows = random.sample(range(n), k)

            selected_ind = her_ind - 1 - np.array(selected_rows)
            selected_ind = selected_ind.astype(np.float)

            selected_ind[selected_ind < 0] += buffer.replay_buffer_size

            her_goals = buffer.buffer[selected_ind.astype(int), 2 * n +
                                      1:3 * n + 1]
            for goal in her_goals:
                replay_buffer_row = np.mod(
                    her_ind - t - 1 + buffer.replay_buffer_size,
                    buffer.replay_buffer_size)

                s = buffer.buffer[int(replay_buffer_row), 0:n]
                a = buffer.buffer[int(replay_buffer_row), 2 * n]
                s_next = buffer.buffer[int(replay_buffer_row), 2 * n +
                                       1:3 * n + 1]

                reward_her = reward_fun(s_next, goal)

                buffer.add(s, goal, a, s_next, reward_her)
        return buffer

    if (method == 'future'):
        return _her_future(buffer, 4, n, reward_fun)
    elif (method == 'final'):
        return _her_final(buffer, n, reward_fun)
