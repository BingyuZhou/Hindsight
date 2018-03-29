import tensorflow as tf
import random
import numpy as np
from bitflipping import bitflipping as bf


class DQN(bf):
    """
    DQN solver
    """

    def __init__(self, eps=1, annealing=0.9, replay_buffer_size=1000, n):
        self.eps = eps  # epsilon-greedy policy
        self.annealing = annealing  # annealing of epsilon
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.n = n  # action space

    def Q_NN(self, hidden_layer):
        """
        Build up the tf.Graph of Q network
        """
        x = tf.placeholder(tf.float32, shape=(None, self.n + 1))
        y = tf.placeholder(tf.float32, shape=(None, 1))

        input = x
        for hid in hidden_layer:
            h = tf.layers.dense(input, hid, activation=tf.nn.tanh)
            input = h
        Q_pred = tf.layers.dense(h, 1)  # output layer

        return Q_pred

    def eps_greedy(self, state, sess, model):
        """
        Epsilon greedy policy
        """
        p = random.random()

        state = np.asarray(state, dtype=int)

        model = tf.estimator.Estimator(
            model_fn=Q_NN, model_dir='/tmp/bitflipping')

        if (p < self.eps):  # random action
            action = random.randint(0, self.n - 1)
            return action
        else:  # greedy policy
            Q_base = 0
            for action in range(self.n):
                X = np.concatenate((state, action))
                Q_pred = sess.run(model, feedict={x: X})

                if Q > Q_base:
                    Q_max = Q
                    action_opt = action
                    Q_base = Q
            return action_opt

    def _init_state(self):
        state = []
        for i in range(self.n):
            if (random.random() < 0.5):
                state.append(1)
            else:
                state.append(0)

        return state

    def tain_Q(self, model, episode, T):
        """
        DQN algorithm for training Q network
        """
        saver = tf.train.Saver()
        replay_buffer = []
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(
                './summary/', sess.graph)  # tensorboard
            init = tf.global_variables_initializer()
            sess.run(init)
            for ep in range(episode):
                s0 = self._init_state()  # initialize state

                for t in range(T):
                    a = self.eps_greedy(s, sess, model)
                    r = bf.bitflipping.reward(s)
                    s_next = bf.bitflipping.update_state(a)

                    replay_buffer.append((s,a,s_next,r))
