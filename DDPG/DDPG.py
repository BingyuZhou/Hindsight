"""
Deep Deterministic Policy Gradient
"""
import tensorflow as tf
import nump as np


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

        self.discount = params['discount']
        self.decay = params['decay']
        self.batch_size = params['batch_size']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']

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
