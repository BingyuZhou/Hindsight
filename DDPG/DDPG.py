"""
Deep Deterministic Policy Gradient
"""
import tensorflow as tf
import numpy as np
from copy import copy
import random


class DDPG():
    def __init__(self, actor, critic, replaybuffer, params, state_shape,
                 action_shape, action_range):
        self.s_0 = tf.placeholder(tf.float32, (None, state_shape), 's_0')
        self.s_1 = tf.placeholder(tf.float32, (None, state_shape), 's_1')
        self.actions = tf.placeholder(tf.float32, (None, action_shape),
                                      'actions')
        self.rewards = tf.placeholder(tf.float32, (None, 1), 'rewards')
        self.critic_target = tf.placeholder(tf.float32, (None, 1),
                                            'critic_target')
        self.terminal = tf.placeholder(tf.float32, (None, 1), 'terminal')

        self.actor = actor
        self.critic = critic
        self.replaybuffer = replaybuffer
        self.critic_Q = critic(self.s_0, self.actions)
        self.actor_A = actor(self.s_0)
        self.critic_with_actor = critic(self.s_0, self.actor_A, reuse=True)
        self.target_actor = copy(actor)
        self.target_actor.name = 'target_actor'
        self.target_critic = copy(critic)
        self.target_critic.name = 'target_critic'
        self.target_critic_Q = self.target_critic(self.s_0, self.actor_A)
        self.target_actor_A = self.target_actor(self.s_0)
        self.target_critic_with_actor = self.target_critic(
            self.s_0, self.target_actor_A, reuse=True)

        self.discount = params['discount']
        self.decay = params['decay']
        self.batch_size = params['batch_size']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.eps = params['eps']
        self.action_range = action_range
        """ L = -E[Q(s, pi(s))]"""
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor)
        self.actor_opt = tf.train.AdamOptimizer(
            learning_rate=self.lr_actor,
            beta1=0.9,
            beta2=0.999,
            name='Adam_actor')
        """ L = E[(Q_pred - Q_target)^2]"""
        self.critic_loss = tf.losses.mean_squared_error(
            self.critic_target,
            self.critic_Q,
            reduction=tf.losses.Reduction.MEAN)
        self.critic_opt = tf.train.AdamOptimizer(
            learning_rate=self.lr_critic,
            beta1=0.9,
            beta2=0.999,
            name='Adam_critic')

        var_list_critic = tf.trainable_variables(scope='critic')
        var_list_actor = tf.trainable_variables(scope='actor')
        global_step = tf.train.get_or_create_global_step()
        self.critic_train_op = self.critic_opt.minimize(
            loss=self.critic_loss,
            global_step=global_step,
            var_list=var_list_critic)
        self.actor_train_op = self.actor_opt.minimize(
            self.actor_loss, global_step=global_step, var_list=var_list_actor)

        # target Q value for critic (y in the paper)
        self.target_Q_op = self.rewards+self.discount * \
            (1.0-self.terminal) * \
            self.target_critic(self.s_1, self.target_actor(self.s_1, reuse=True), reuse=True)

    def _initialize(self, sess):
        self.sess = sess
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):
        """ One step optimization of the Actor
        and Critic network"""
        batch = self.replaybuffer.sample(self.batch_size)

        target_Q = self.sess.run(
            self.target_Q_op,
            feed_dict={
                self.rewards: batch['rewards'],
                self.terminal: batch['terminal'],
                self.s_1: batch['s1']
            })

        _, _, critic_loss, actor_loss = self.sess.run(
            [
                self.critic_train_op, self.actor_train_op, self.critic_loss,
                self.actor_loss
            ],
            feed_dict={
                self.s_0: batch['s0'],
                self.actions: batch['a'],
                self.critic_target: target_Q
            })

        return critic_loss, actor_loss

    def update_target_nn(self):
        """Update the target networks to slowly track the current
        optimized Actor and Critic respectively
        """
        tf.logging.info('---Start updating target nets---')
        update_target_critic_op = []
        update_target_actor_op = []

        vars_critic = self.critic.trainable_var()
        vars_critic_target = self.target_critic.trainable_var()
        vars_actor = self.actor.trainable_var()
        vars_actor_target = self.target_actor.trainable_var()
        assert len(vars_critic) == len(vars_critic_target)
        assert len(vars_actor) == len(vars_actor_target)
        for var, var_target in zip(vars_critic, vars_critic_target):
            tf.logging.info('{} -> {}'.format(var.name, var_target.name))
            update_target_critic_op.append(
                tf.assign(var_target,
                          (1.0 - self.decay) * var + self.decay * var_target))

        for var, var_target in zip(vars_actor, vars_actor_target):
            tf.logging.info('{} -> {}'.format(var.name, var_target.name))
            update_target_actor_op.append(
                tf.assign(var_target,
                          (1.0 - self.decay) * var + self.decay * var_target))

        self.sess.run(update_target_critic_op)
        self.sess.run(update_target_actor_op)

    def eps_policy(self, state):
        """ Epsilon-greedy policy"""
        if (random.random() < self.eps):
            action = random.uniform(0, 1)
            action = np.array(action)
        else:
            action = self.sess.run(self.actor_A, feed_dict={self.s_0: state})
        return action

    def pi(self, state, eps_greedy=False, compute_V=False, using_target=False):
        """ a = actor(state)
        Compute the optimal action from Actor network,
        it is also able to compute the Value of state from
        Critic netwoek
        """
        if eps_greedy:
            action = self.eps_policy(state)
            if compute_V:
                if using_target:
                    Q = self.sess.run(
                        self.target_critic_Q,
                        feed_dict={
                            self.s_0:
                            state,
                            self.actions:
                            np.asarray(action, dtype=np.float32).reshape((-1,
                                                                          1))
                        })
                else:
                    Q = self.sess.run(
                        self.critic_Q,
                        feed_dict={
                            self.s_0:
                            state,
                            self.actions:
                            np.asarray(action, dtype=np.float32).reshape((-1,
                                                                          1))
                        })
            else:
                Q = None
        else:
            if compute_V:
                if using_target:
                    action, Q = self.sess.run(
                        [self.target_actor_A, self.target_critic_with_actor],
                        feed_dict={self.s_0: state})
                else:
                    action, Q = self.sess.run(
                        [self.actor_A, self.critic_with_actor],
                        feed_dict={self.s_0: state})
            else:
                if using_target:
                    action = self.sess.run(
                        self.target_actor_A, feed_dict={self.s_0: state})
                else:
                    action = self.sess.run(
                        self.actor_A, feed_dict={self.s_0: state})
                Q = None
        return action, Q

    def debug(self):
        """ Add parameters to tensorboard for monitoring"""

        tf.summary.scalar('critic_loss', self.critic_loss)
        tf.summary.scalar('actor_loss', self.actor_loss)
