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
        self.target_actor = copy(actor)
        self.target_critic = copy(critic)

        self.discount = params['discount']
        self.decay = params['decay']
        self.batch_size = params['batch_size']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.eps = params['eps']
        self.action_range = action_range

    def _initialize(self, sess):
        self.sess = sess

    def _actor_loss(self, critic_with_actor):
        """ L = -E[Q(s, pi(s))]"""
        self.actor_loss = -tf.reduce_mean(critic_with_actor)

    def _actor_opt(self):
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=self.lr_actor)

    def _critic_loss(self):
        """ L = E[(Q_pred - Q_target)^2]"""
        self.critic_loss = tf.losses.mean_squared_error(
            self.critic_target,
            self.critic_Q,
            reduction=tf.losses.Reduction.MEAN)

    def _critic_opt(self):
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=self.lr_critic)

    def train(self, global_step):
        """ One step optimization of the Actor
        and Critic network"""
        batch = self.replaybuffer.sample(self.batch_size)

        # target Q value for critic (y in the paper)
        target_Q_op = self.rewards+self.discount * \
            (1.0-self.terminal) * \
            self.target_critic(self.s_1, self.target_actor(self.s_1))

        target_Q = self.sess.run(
            target_Q_op,
            feed_dict={
                self.rewards: batch['rewards'],
                self.terminal: batch['terminal'],
                self.s_1: batch['s1']
            })

        var_list_critic = tf.trainable_variables(scope='critic')
        var_list_actor = tf.trainable_variables(scope='actor')
        critic_train_op = self.critic_opt.minimize(
            loss=self.critic_loss,
            global_step=global_step,
            var_list=var_list_critic)
        actor_train_op = self.actor_opt.minimize(
            self.actor_loss, global_step=global_step, var_list=var_list_actor)

        critic_loss, actor_loss = self.sess.run(
            [critic_train_op, actor_train_op],
            feed_dict={
                self.s_0: batch['s0'],
                self.actions: batch['a'],
                self.critic_target: target_Q
            })

        return critic_loss, actor_loss

    def update_target_nn(self, vars, vars_target):
        """Update the target networks to slowly track the current
        optimized Actor and Critic respectively
        """
        tf.logging.info('---Start updating target nets---')
        update_target_op = []
        assert len(vars) == len(vars_target)
        for var, var_target in zip(vars, vars_target):
            tf.logging.info('{} -> {}'.format(var.name, var_target.name))
            update_target_op.append(
                tf.assign(var_target,
                          (1.0 - self.decay) * var + self.decay * var))

        self.sess.run(update_target_op)

    def eps_policy(self, state):
        """ Epsilon-greedy policy"""
        if (random.random() < self.eps):
            action = random.randint(self.action_range[0], self.action_range[1])
        else:
            action = self.sess.run(self.actor_A, feed_dict={self.s_0: state})
        return action

    def pi(self, state, eps_greedy=False, compute_V=False):
        """ a = actor(state)
        Compute the optimal action from Actor network,
        it is also able to compute the Value of state from
        Critic netwoek
        """
        if eps_greedy:
            action = self.eps_policy(state)
            if compute_V:
                Q = self.sess.run(
                    self.critic,
                    feed_dict={
                        self.s_0: state,
                        self.actions: action
                    })
            else:
                Q = None
        else:
            if compute_V:
                action, Q = self.sess.run(
                    [self.actor_A,
                     self.critic(self.s_0, self.actor_A)],
                    feed_dict={self.s_0: state})
            else:
                action = self.sess.run(
                    self.actor_A, feed_dict={self.s_0: state})
                Q = None
        return action, Q

    def debug(self):
        """ Add parameters to tensorboard for monitoring"""

        tf.summary.scalar('critic_loss', self.critic_loss)
        tf.summary.scalar('actor_loss', self.actor_loss)
