"""
Deep Deterministic Policy Gradient
"""
import tensorflow as tf
import numpy as np
from copy import copy


class DDPG():
    def __init__(self, actor, critic, replaybuffer, params, state_shape, action_shape):
        self.s_0 = tf.placeholder(tf.float32, (None, state_shape), 's_0')
        self.s_1 = tf.placeholder(tf.float32, (None, state_shape), 's_1')
        self.actions = tf.placeholder(
            tf.float32, (None, action_shape), 'actions')
        self.rewards = tf.placeholder(tf.float32, (None, 1), 'rewards')
        self.critic_target = tf.placeholder(
            tf.float32, (None, 1), 'critic_target')
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

    def _actor_loss(self, critic_with_actor):
        """ L = -E[Q(s, pi(s))]"""
        self.actor_loss = -tf.reduce_mean(critic_with_actor)

    def _actor_opt(self):
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=self.lr_actor)

    def _critic_loss(self):
        """ L = E[(Q_pred - Q_target)^2]"""
        self.critic_loss = tf.losses.mean_squared_error(
            self.critic_target, self.critic_Q, reduction=tf.losses.Reduction.MEAN)

    def _critic_opt(self):
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=self.lr_critic)

    def train(self):
        """ One step optimization of the Actor
        and Critic network"""

    def update_target_nn(self):
        """Update the target networks to slowly track the current
        optimized Actor and Critic respectively
        """

    def pi(self, state, compute_V=False):
        """ a = actor(state)
        Compute the optimal action from Actor network,
        it is also able to compute the Value of state from
        Critic netwoek
        """
