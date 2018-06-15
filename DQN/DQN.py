import tensorflow as tf
import random
import numpy as np
from bitflipping import bitflipping as bf


class DQN:
    """
    DQN solver
    """

    def __init__(self,
                 x,
                 hid,
                 n,
                 discount=0.98,
                 eps=1,
                 annealing=0.9,
                 tau=0.95,
                 replay_buffer_size=1000,
                 batch_size=124):
        self.eps = eps  # epsilon-greedy policy
        self.annealing = annealing  # annealing of epsilon
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.batch_size = batch_size
        self.n = n  # action space
        self.discount = discount
        self.tau = tau
        self.model = self.Q_NN(x, hid, True, "model")
        # Target NN used to compute targets for the sake of training stability
        self.targetModel = self.Q_NN(x, hid, False, "targetmodel")

    def Q_NN(self, x, hidden_layer, is_training, scope):
        """
        Build up the model of Q-NN
        - x: {state, goal}
        - hidden_layer: units at each layer
        - Output: Q value for all possible actions
        """

        input = x
        # Xavier initializer
        with tf.variable_scope(scope):
            init = tf.contrib.layers.xavier_initializer()
            for hid in hidden_layer:
                h = tf.layers.dense(
                    input,
                    hid,
                    activation=tf.nn.relu,
                    kernel_initializer=init,
                    bias_initializer=tf.zeros_initializer(),
                    trainable=is_training,
                    reuse=False)
                input = h
            Q_pred = tf.layers.dense(
                h,
                self.n,
                kernel_initializer=init,
                bias_initializer=tf.zeros_initializer(),
                trainable=is_training,
                reuse=False)  # output layer

        return Q_pred

    def V_value(self, sess, state, goal, x):
        """
        V = max_a Q(s,a)
        """
        Q_max = []

        X = np.concatenate(
            (state.reshape((1, -1)), goal.reshape((1, -1))), axis=1)
        Q_target_nn = sess.run(self.targetModel, feed_dict={x: X})

        action_opt = np.argmax(Q_target_nn)
        Q_max = Q_target_nn[:, action_opt]

        return Q_max, action_opt

    def eps_greedy(self, global_i, state, goal, sess, x):
        """
        Epsilon greedy policy
        """
        p = random.random()

        eps_current = max(0.1, 1 - 4e-4 * global_i)

        if (p < eps_current):  # random action
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
        model_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
        target_var_list = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="targetmodel")
        # n = len(var_list)
        op = []  # operation to transfer weights and bias
        # for i in range(n // 2):
        #     # decay coefficients
        #     decay_value = (
        #         1.0 - self.tau) * var_list[i] + self.tau * var_list[i + n // 2]
        #     op.append(var_list[i + n // 2].assign(decay_value))
        for i, var in enumerate(target_var_list):
            decay_value = (1.0 - self.tau) * model_var_list[i] + self.tau * var
            op.append(var.assign(decay_value))
        sess.run(op)

    def train_Q(self, x, y, epoch, cycles, episode, T, iteration):
        """
        DQN algorithm for training Q network
        """

        action = tf.placeholder(tf.int32, shape=None)

        a_onehot = tf.one_hot(action, self.n)

        replay_buffer = np.array([]).reshape((-1, self.n * 3 + 2))

        # Loss

        loss = tf.losses.mean_squared_error(
            y,
            tf.reduce_sum(tf.multiply(self.model, a_onehot), axis=1),
            reduction=tf.losses.Reduction.MEAN)
        # loss = tf.clip_by_value(
        #     (y - tf.reduce_sum(tf.multiply(self.model, a_onehot), axis=1)), -1,
        #     1)
        # loss = tf.losses.mean_squared_error(
        #     0, loss, reduction=tf.losses.Reduction.MEAN)
        loss_tb = tf.summary.scalar('loss', loss)

        W1 = tf.trainable_variables('model/dense/kernel:0')
        W1_target = tf.global_variables('targetmodel/dense/kernel:0')
        W1_tb = tf.summary.histogram('W1', W1[0])
        W1_target_tb = tf.summary.histogram('W2', W1_target[0])

        success_rate = tf.Variable(
            initial_value=0.0, name='success_rate', trainable=False)
        success_tb = tf.summary.scalar('success_rate', success_rate)

        merge_tb = tf.summary.merge_all()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(loss)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./summary/',
                                           sess.graph)  # tensorboard
            init = tf.global_variables_initializer()
            sess.run(init)
            losses = []
            success_all = []

            global_i = 0

            self.update_target_model(sess)
            for e in range(epoch):

                for cycle in range(cycles):
                    success = 0
                    for ep in range(episode):

                        # initialize a bitflipping env
                        s = self._sample_state()  # initialize state
                        goal = self._sample_state()  # goal
                        while np.array_equal(s, goal):
                            goal = self._sample_state()

                        env = bf(s, goal, self.n)  # bitflipping env

                        # Sample training data set
                        for t in range(T):
                            if np.array_equal(s, goal):
                                success += 1
                                break

                            a = self.eps_greedy(global_i, s, goal, sess, x)

                            s_next = env.update_state(a)
                            r = env.reward(s_next)

                            # Put (s, g, a, s', r) into the replay buffer
                            if (replay_buffer.shape[0] <
                                    self.replay_buffer_size):
                                replay_buffer = np.append(
                                    replay_buffer,
                                    np.concatenate(
                                        (s.reshape(
                                            (1, -1)), goal.reshape(
                                                (1, -1)), np.array([[a]]),
                                         s_next.reshape(
                                             (1, -1)), np.array([[r]])),
                                        axis=1),
                                    axis=0)
                            else:
                                replay_buffer = np.delete(
                                    replay_buffer, 0, axis=0)
                                replay_buffer = np.append(
                                    replay_buffer,
                                    np.concatenate(
                                        (s.reshape(
                                            (1, -1)), goal.reshape(
                                                (1, -1)), np.array([[a]]),
                                         s_next.reshape(
                                             (1, -1)), np.array([[r]])),
                                        axis=1),
                                    axis=0)

                            s = np.copy(s_next)
                        global_i += 1
                    success_rate_op = success_rate.assign(success / episode)

                    # end of experience replay

                    # One step optimization of Q neural network
                    for t in range(iteration):
                        # Sample random minibatches from the replay buffer to update Q-network
                        # use half of replay buffer to do minibatch gradient descent
                        if (replay_buffer.shape[0] > self.batch_size):
                            mini_batch_index = np.random.choice(
                                replay_buffer.shape[0],
                                self.batch_size,
                                replace=False)
                        else:
                            mini_batch_index = np.random.choice(
                                replay_buffer.shape[0],
                                self.batch_size,
                                replace=True)

                        batch = replay_buffer[mini_batch_index]

                        # print(batch)

                        # True Q values
                        Q_true = np.zeros((self.batch_size, 1))

                        for i in range(self.batch_size):
                            next_state = batch[i, 2 * self.n + 1:3 * self.n +
                                               1]
                            # if next state is goal state
                            if np.array_equal(next_state, goal):
                                Q_true_i = batch[i, -1]
                            else:
                                V, _ = self.V_value(sess, next_state, goal, x)
                                # Bellman equation
                                Q_true_i = np.clip(
                                    batch[i, -1] + self.discount * V,
                                    -1.0 / (1.0 - self.discount), 0)

                            Q_true[i] = Q_true_i

                        # Update Q-network with the sampled batch data

                        input = batch[:, 0:2 * self.n]
                        ls, _, _, summary = sess.run(
                            [loss, train_step, success_rate_op, merge_tb],
                            feed_dict={
                                x: input,
                                y: Q_true,
                                action: batch[:, 2 * self.n]
                            })

                        losses.append(ls)
                        writer.add_summary(
                            summary, e * cycles * episode * iteration +
                            cycle * episode * iteration + ep * iteration + t)
                    # end of optimization

                    # Update target model every certain steps
                    self.update_target_model(sess)
                    print('Epoch {0} Cycle {1} Episode {2}: loss is {3:.3g}'.
                          format(e, cycle, ep, ls))

            writer.close()
            saver = tf.train.Saver()
            saver.save(sess, '/tmp/model.ckpt')  # save model variables
        return losses, success_all
