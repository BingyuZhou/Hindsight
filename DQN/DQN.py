import tensorflow as tf
import random
import numpy as np
from bitflipping import bitflipping as bf


class DQN:
    """
    DQN solver
    """

    def __init__(self, n, discount=0.98, eps=1, annealing=0.9, replay_buffer_size=1000):
        self.eps = eps  # epsilon-greedy policy
        self.annealing = annealing  # annealing of epsilon
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.n = n  # action space
        self.discount = discount

    def Q_NN(self, x, hidden_layer):
        """
        Build up the tf.Graph of Q network
        """

        input = x
        for hid in hidden_layer:
            h = tf.layers.dense(input, hid, activation=tf.nn.tanh)
            input = h
        Q_pred = tf.layers.dense(h, 1)  # output layer

        return Q_pred

    def run_model(self, session, Q_pred, loss, train_step, x, Q_true, is_training):
        """
        Compute the tf.Graph
        """

        if is_training:
            # Update Q-network
            ls = session.run([loss, train_step], feedict={x: x, y: Q_true})
        else:
            ls = session.run(loss, feedict={x: x, y: Q_true})

        return ls

    def V_value(self, sess, state, model):
        """
        V = max_a Q(s,a)
        """
        Q_base = 0
        for action in range(self.n):
            X = np.concatenate((state, action))
            Q_pred = sess.run(model, feedict={x: X, y: y})

            if Q > Q_base:
                Q_max = Q
                action_opt = action
                Q_base = Q

        return Q_max, action_opt

    def eps_greedy(self, state, sess, model):
        """
        Epsilon greedy policy
        """
        p = random.random()

        state = np.asarray(state, dtype=int)

        if (p < self.eps):  # random action
            action = random.randint(0, self.n - 1)
            return action
        else:  # greedy policy
            _, action_opt = self.V_value(sess, state, model)
            return action_opt

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

    def train_Q(self, episode, T):
        """
        DQN algorithm for training Q network
        """
        saver = tf.train.Saver()
        replay_buffer = np.array([]).reshape((-1, self.n*2+2))
        rb_ind = 0  # index used for replay_buffer

        x = tf.placeholder(tf.float32, shape=(None, self.n + 1))
        y = tf.placeholder(tf.float32, shape=(None, 1))
        hid = [20, 20]

        # Predicted Q values
        Q_pred = self.Q_NN(x, hid)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(
                './summary/', sess.graph)  # tensorboard
            init = tf.global_variables_initializer()
            sess.run(init)
            losses = []
            for ep in range(episode):
                # initialize a bitflipping env
                s = self._sample_state()  # initialize state
                goal = self._sample_state()  # goal
                env = bf(s, goal, self.n)  # bitflipping env

                for t in range(T):
                    a = self.eps_greedy(s, sess, model)
                    r = env.reward(s)
                    s_next = env.update_state(a)

                    if (replay_buffer.shape[0] < self.replay_buffer_size):
                        replay_buffer = np.append(replay_buffer, np.concatenate(
                            (s, np.array([[a]]), s_next, np.array([[r]]))), axis=0)
                    else:
                        replay_buffer[rb_ind, :] = np.concatenate(
                            (s, np.array([[a]]), s_next, np.array([[r]])))
                        rb_ind = (rb_ind+1) % self.replay_buffer_size

                    # Sample random minibatches from the replay buffer to update Q-network
                    # use half of replay buffer to do minibatch gradient descent
                    batch_size = replay_buffer.shape[0] / 2
                    mini_batch_index = np.random.choice(
                        replay_buffer.shape[0], batch_size, replace=False)

                    batch = replay_buffer(mini_batch_index)

                    # True Q values
                    Q_true = np.zeros((batch.shape[0],))

                    for i in range(batch.shape[0]):
                        next_state = batch[i, self.n+1: 2*self.n+1]
                        # if next state is goal state
                        if np.array_equal(next_state, goal):
                            Q_true_i = batch[i, -1]
                        else:
                            V, _ = self.V_value(sess, next_state, model)
                            # Bellman equation
                            Q_true_i = batch[i, -1]+self.discount * V
                        Q_true[i] = Q_true_i

                    # Loss
                    loss = tf.losses.mean_squared_error(y, Q_pred)

                    # Optimizer
                    optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate=0.01)
                    train_step = optimizer.minimize(loss)

                    # Update Q-network with the sampled batch data
                    ls = self.run_model(
                        sess, Q_pred, loss, train_step, batch[:, 0:self.n+1], Q_true, True)

                    losses.append(ls)
                    print('Episode {0}: loss is {1:.3g}'.format(ep, ls))

            writer.close()
            saver.save(sess, '/tmp/model.ckpt')  # save model variables
        return losses
