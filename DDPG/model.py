"""
Architecture of Actor and Critic NN
"""
import tensorflow as tf
import numpy as np


class Actor:
    def __init__(self, hid_layer, batch_norm=False):
        self.hid_layer = hid_layer
        self.batch_norm = batch_norm

    def __call__(self, state):
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

        output = tf.layers.dense(input)
