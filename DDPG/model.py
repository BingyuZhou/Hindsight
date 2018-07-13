"""
Architecture of Actor and Critic NN
"""
import tensorflow as tf
import numpy as np


class Actor:
    def __init__(
            self,
            hid_layer,
            action_shape,
            name,
            batch_norm=False,
    ):
        self.hid_layer = hid_layer
        self.batch_norm = batch_norm
        self.action_shape = action_shape
        self.name = name

    def __call__(self, state):

        with tf.variable_scope(self.name):
            input = state

            if self.batch_norm:
                for hid in self.hid_layer:
                    x = tf.layers.dense(input, hid, activation=None)
                    x = tf.layers.batch_normalization(x)
                    input = tf.nn.relu(x)
            else:
                for hid in self.hid_layer:
                    x = tf.layers.dense(input, hid, activation=tf.nn.relu)
                    input = x
            # Directly compute action
            output = tf.layers.dense(
                input, self.action_shape, activation=tf.nn.sigmoid)
        return output

    def trainable_var(self):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.name)

        return var_list


class Critic:
    def __init__(self, hid_layer, name, batch_norm=False):
        self.hid_layer = hid_layer
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, state, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.concat([state, action], 1)

            if self.batch_norm:
                for hid in self.hid_layer:
                    x = tf.layers.dense(x, hid, activation=None)
                    x = tf.layers.batch_normalization(x)
                    x = tf.nn.relu(x)
            else:
                for hid in self.hid_layer:
                    x = tf.layers.dense(x, hid, activation=tf.nn.relu)

            # Predicted Q value
            output = tf.layers.dense(x, 1, activation=None)

        return output

    def trainable_var(self):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.name)

        return var_list
