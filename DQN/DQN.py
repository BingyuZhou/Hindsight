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
                 tau=0.95,
                 replay_buffer_size=1000,
                 batch_size=128):
        self.eps = eps  # epsilon-greedy start value
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
                    activation=None,
                    kernel_initializer=init,
                    bias_initializer=tf.zeros_initializer(),
                    trainable=is_training)
                input = h
            Q_pred = tf.layers.dense(
                h,
                self.n,
                kernel_initializer=init,
                bias_initializer=tf.zeros_initializer(),
                trainable=is_training)  # output layer

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

        eps_current = max(0.05, 1 - 2e-3 * global_i)

        if (p < eps_current):  # random action
            action = random.randint(0, self.n - 1)
            return action
        else:  # greedy policy
            _, action_opt = self.V_value(sess, state, goal, x)
            return action_opt

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

    def train_Q(self, x, y, epoch, cycles, episode, iteration):
        """
        DQN algorithm for training Q network
        """

        action = tf.placeholder(tf.int32, shape=None)

        a_onehot = tf.one_hot(action, self.n)

        replay_buffer = np.array([]).reshape((-1, self.n * 3 + 2))
        rb_ind = 0

        # Loss
        Q_pred = tf.reduce_sum(self.model * a_onehot, axis=1)
        errors = tf.losses.huber_loss(y, Q_pred)
        loss = tf.reduce_mean(errors)
        # loss = tf.losses.mean_squared_error(
        #     y,
        #     tf.reduce_sum(tf.multiply(self.model, a_onehot), axis=1),
        #     reduction=tf.losses.Reduction.MEAN)
        # loss = tf.clip_by_value(
        #     (y - tf.reduce_sum(tf.multiply(self.model, a_onehot), axis=1)), -1,
        #     1)
        # loss = tf.losses.mean_squared_error(
        #     0, loss, reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar('loss', loss)

        W1 = tf.trainable_variables('model/dense/kernel:0')
        W1_target = tf.global_variables('targetmodel/dense/kernel:0')
        tf.summary.histogram('W1', W1[0])
        tf.summary.histogram('W2', W1_target[0])

        success_rate = tf.Variable(
            initial_value=0.0, name='success_rate', trainable=False)
        tf.summary.scalar('success_rate', success_rate)

        merge_tb = tf.summary.merge_all()

        # Optimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=global_step,
            decay_steps=500,
            decay_rate=0.98,
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

        merge_tb = tf.summary.merge_all()
        global_i = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./summary/',
                                           sess.graph)  # tensorboard
            init = tf.global_variables_initializer()
            sess.run(init)
            losses = []
            success_all = []
            success = 0

            self.update_target_model(sess)
            for e in range(epoch):
                for cycle in range(cycles):
                    success = 0
                    for ep in range(episode):

                        # initialize a bitflipping env

                        env = bf(self.n)  # bitflipping env

                        # Sample training data set
                        for t in range(self.n):

                            s = np.copy(env.state)
                            a = self.eps_greedy(global_i, s, env.goal, sess, x)

                            s_next = env.update_state(a)
                            r = env.reward(s_next)

                            # Put (s, g, a, s', r) into the replay buffer
                            if (replay_buffer.shape[0] <
                                    self.replay_buffer_size):
                                replay_buffer = np.append(
                                    replay_buffer,
                                    np.concatenate(
                                        (s.reshape(
                                            (1, -1)), env.goal.reshape(
                                                (1, -1)), np.array([[a]]),
                                         s_next.reshape(
                                             (1, -1)), np.array([[r]])),
                                        axis=1),
                                    axis=0)
                            else:
                                replay_buffer[rb_ind, :] = np.concatenate(
                                    (s.reshape(
                                        (1, -1)), env.goal.reshape((1, -1)),
                                     np.array([[a]]), s_next.reshape(
                                         (1, -1)), np.array([[r]])),
                                    axis=1)
                                rb_ind = (rb_ind + 1) % self.replay_buffer_size

                            if (r == 0):
                                success += 1
                                break

                        global_i += 1
                        success_rate_op = success_rate.assign(
                            success / episode)
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
                            reward = batch[i, -1]
                            goal = batch[i, self.n:2 * self.n]
                            next_state = batch[i, 2 * self.n + 1:3 * self.n +
                                               1]
                            # if next state is goal state
                            if reward == 0:
                                Q_true_i = reward
                            else:
                                V, _ = self.V_value(sess, next_state, goal, x)
                                # Bellman equation
                                Q_true_i = np.clip(
                                    reward + self.discount * V,
                                    -1.0 / (1.0 - self.discount), 0)

                            Q_true[i] = Q_true_i
                        # print(Q_true)

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
                        writer.add_summary(summary, global_step.eval())
                    # end of optimization
                    print('Epoch {0} Cycle {1}: loss is {2:.3g}'.format(
                        e, cycle, ls))
                    # Update target model every certain steps
                    if (cycle % 2 == 0):
                        self.update_target_model(sess)

            writer.close()
            saver = tf.train.Saver()
            saver.save(sess, '/tmp/model.ckpt')  # save model variables
        return losses, success_all
