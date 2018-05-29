import tensorflow as tf
import random
import numpy as np
from bitflipping import bitflipping as bf


class DQN(bf):
    """
    DQN solver
    """

    def __init__(self, eps=1, annealing=0.9, replay_buffer_size=1000, n, discount=0.98):
        self.eps = eps  # epsilon-greedy policy
        self.annealing = annealing  # annealing of epsilon
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.n = n  # action space
        self.discount = discount

    def Q_NN(self, x, y, hidden_layer):
        """
        Build up the tf.Graph of Q network
        """
        # x = tf.placeholder(tf.float32, shape=(None, self.n + 1))
        # y = tf.placeholder(tf.float32, shape=(None, 1))

        input = x
        for hid in hidden_layer:
            h = tf.layers.dense(input, hid, activation=tf.nn.tanh)
            input = h
        Q_pred = tf.layers.dense(h, 1)  # output layer

        return Q_pred

    def V_value(self, state, model):
        Q_base = 0
        for action in range(self.n):
            X = np.concatenate((state, action))
            Q_pred = sess.run(model, feedict={x: X})

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
            _, action_opt = V_value(state, model)
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
        replay_buffer = np.array([]).reshape((-1, self.n*2+2))
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

                    replay_buffer = np.append(replay_buffer, [np.concatenate(
                        (s, np.array([[a]]), s_next, np.array([[r]]))], axis=0)

                    # Sample random minibatches from the replay buffer to update Q-network
                    # use half of replay buffer to do minibatch gradient descent
                    batch_size=replay_buffer.shape[0] / 2
                    mini_batch_index=np.random.choice(
                        replay_buffer.shape[0], batch_size, replace=False)

                    batch=replay_buffer(mini_batch_index)

                    # Predicted Q values
                    
                    Q_true = np.zeros((batch.shape[0],))
                    # True Q values
                    for i in range(batch.shape[0]):
                        next_state = batch[i, self.n+1 : 2*self.n+1]
                        if np.array_equal(next_state, self.goal): # if next state is goal state
                            Q_true_i = batch[i, -1]
                        else:
                            V, _ = V_value(next_state, model)
                            Q_true_i=batch[i, -1]+self.discount * V # Bellman equation
                        Q_true[i] = Q_true_i

                    Q_pred = Q_NN(batch[:, 0:n+1], Q_true, hidden_layer=hid)
                    
                    # Loss 
                    loss = tf.loss.mean_squared_error(Q_true, Q_pred)

                    # Optimizer
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                    train_step = optimizer.minimize(loss)

                    # Update Q-network
                    ls = session.run(loss)
