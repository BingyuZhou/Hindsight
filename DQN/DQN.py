import tensorflow as tf
import random
import numpy as np
from bitflipping import bitflipping as bf


class DQN:
    """
    DQN solver
    """

    def __init__(self, x, hid,  n, discount=0.98, eps=1, annealing=0.9, replay_buffer_size=1000):
        self.eps = eps  # epsilon-greedy policy
        self.annealing = annealing  # annealing of epsilon
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.n = n  # action space
        self.discount = discount
        self.model = self.Q_NN(x, hid)
        # Target NN used to compute targets for the sake of training stability
        self.targetModel = self.Q_NN(x, hid)

    def Q_NN(self, x, hidden_layer):
        """
        Build up the model of Q-NN
        - x: {state, action, goal}
        - hidden_layer: units at each layer
        - trainable: Ture: training model; False: Target model
        - weights: Collected weights for target model
        - bias: Collected bias for target model
        """

        input = x

        for hid in hidden_layer:
            h = tf.layers.dense(input, hid, activation=tf.nn.tanh)
            input = h
        Q_pred = tf.layers.dense(h, 1)  # output layer

        return Q_pred

    def run_model(self, session, loss, train_step, merged_tb, x, y, X, Q_true, is_training):
        """
        Compute the tf.Graph
        """

        if is_training:
            # Update Q-network
            ls, _, summary = session.run([loss, train_step, merged_tb],
                                         feed_dict={x: X, y: Q_true})
        else:
            ls, summary = session.run(
                [loss, merged_tb], feed_dict={x: X, y: Q_true})

        return ls, summary

    def V_value(self, sess, state, goal, x):
        """
        V = max_a Q(s,a)
        """
        Q_base = -100
        Q_max = None
        for action in range(self.n):
            X = np.concatenate((state, np.array([action]), goal))
            Q_pred = sess.run(self.targetModel, feed_dict={x: X.reshape(
                (1, -1))})

            if Q_pred > Q_base:
                Q_max = Q_pred
                action_opt = action
                Q_base = Q_pred

        return Q_max, action_opt

    def eps_greedy(self, state, goal, sess, x):
        """
        Epsilon greedy policy
        """
        p = random.random()

        if (p < self.eps):  # random action
            action = random.randint(0, self.n - 1)
            return action
        else:  # greedy policy
            _, action_opt = self.V_value(sess, state, goal, x)
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

    def update_target_model(self, sess):
        """
        Update target model
        """
        var_list = tf.trainable_variables()
        n = len(var_list)
        op = []  # operation to transfer weights and bias
        for i in range(n//2):
            op.append(var_list[i+n//2].assign(var_list[i]))
        sess.run(op)

    def train_Q(self, x, y,  episode, T, update_step):
        """
        DQN algorithm for training Q network
        """
        saver = tf.train.Saver()
        replay_buffer = np.array([]).reshape((-1, self.n*3+2))
        rb_ind = 0  # index used for replay_buffer

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(
                './summary/', sess.graph)  # tensorboard
            init = tf.global_variables_initializer()
            sess.run(init)
            losses = []

            # Loss
            loss = tf.losses.mean_squared_error(y, self.model)
            loss_tb = tf.summary.scalar('loss', loss)

            W1 = tf.trainable_variables('dense/kernel:0')
            W1_target = tf.trainable_variables('dense_2/kernel:0')
            W1_tb = tf.summary.histogram('W1', W1[0])
            W1_target_tb = tf.summary.histogram('W2', W1_target[0])

            merge_tb = tf.summary.merge_all()

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.01)
            train_step = optimizer.minimize(loss)

            for ep in range(episode):
                # initialize a bitflipping env
                s = self._sample_state()  # initialize state
                goal = self._sample_state()  # goal
                while np.array_equal(s, goal):
                    goal = self._sample_state()

                env = bf(s, goal, self.n)  # bitflipping env

                # Update target model every certain steps
                if (ep % update_step == 0):
                    self.update_target_model(sess)

                for t in range(T):
                    a = self.eps_greedy(s, goal, sess, x)
                    s_next = env.update_state(a)
                    r = env.reward(s_next)

                    # Put (s, a, s', r, g) into the replay buffer
                    if (replay_buffer.shape[0] < self.replay_buffer_size):
                        replay_buffer = np.append(replay_buffer, np.concatenate(
                            (s.reshape((1, -1)), np.array([[a]]), s_next.reshape((1, -1)), np.array([[r]]), goal.reshape((1, -1))), axis=1), axis=0)
                    else:
                        replay_buffer[rb_ind, :] = np.concatenate(
                            (s.reshape((1, -1)), np.array([[a]]), s_next.reshape((1, -1)), np.array([[r]]), goal.reshape((1, -1))), axis=1)
                        rb_ind = (rb_ind+1) % self.replay_buffer_size

                    s = np.copy(s_next)

                    # Sample random minibatches from the replay buffer to update Q-network
                    # use half of replay buffer to do minibatch gradient descent
                    batch_size = max(1, int(replay_buffer.shape[0]))
                    mini_batch_index = np.random.choice(
                        replay_buffer.shape[0], batch_size, replace=False)

                    batch = replay_buffer[mini_batch_index]

                    print(batch)

                    # True Q values
                    Q_true = np.zeros((batch_size, 1))

                    for i in range(batch_size):
                        next_state = batch[i, self.n+1: 2*self.n+1]
                        # if next state is goal state
                        if np.array_equal(next_state, goal):
                            Q_true_i = batch[i, -1]
                        else:
                            V, _ = self.V_value(
                                sess, next_state, goal, x)
                            # Bellman equation
                            Q_true_i = batch[i, -1]+self.discount * V
                        Q_true[i] = Q_true_i

                    # Update Q-network with the sampled batch data

                    batch_goal = batch[:, 2*self.n+2:]

                    input = np.concatenate(
                        (batch[:, 0:self.n+1], batch_goal), axis=1)
                    ls, summary = self.run_model(
                        sess, loss, train_step, merge_tb, x, y, input, Q_true, True)

                    losses.append(ls)
                    writer.add_summary(summary, ep*self.n+t)
                    print('Episode {0}: loss is {1:.3g}'.format(ep, ls))

                    if np.array_equal(s_next, goal):
                        break

            writer.close()
            saver.save(sess, '/tmp/model.ckpt')  # save model variables
        return losses
