import tensorflow as tf
from tensorflow.keras.layers import *

"""
Layers

This is only for implementing layers.
You should not import class or functions from modules or models
"""


def conv2d_bn(filters,
              kernel_size, 
              strides=(1, 1), 
              padding='same', 
              groups=1,
              use_bias=True, 
              kernel_regularizer=None, 
              activation=None, 
              bn_args=None):
    if bn_args is None:
        bn_args = {} # you can put axis, momentum, epsilon in bn_args

    def _conv2d_layer(inputs):
        x = Conv2D(filters, kernel_size, 
                   strides=strides, 
                   padding=padding, 
                   groups=groups,
                   use_bias=use_bias, 
                   kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization(**bn_args)(x)
        if activation:
            x = Activation(activation)(x)
        return x

    return _conv2d_layer


def force_1d_inputs():
    def force(inputs):
        x = inputs
        if len(x.shape) == 4:
            x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)
        return x
    return force


'''
POSITIONAL ENCODINGS
'''
def basic_pos_encoding(input_shape):
    # basic positional encoding from transformer
    k = input_shape[-1] // 2
    w = tf.reshape(tf.pow(10000, -tf.range(k)/k), (1, 1, -1))
    w = tf.constant(tf.cast(w, tf.float32))

    def pos_encoding(inputs):
        assert len(inputs.shape) == 3

        time = tf.shape(inputs)[-2]
        encoding = tf.reshape(tf.range(time, dtype=inputs.dtype), (1, -1, 1))
        encoding = tf.concat([tf.cos(w * encoding), tf.sin(w * encoding)], -1)
        return encoding
    return pos_encoding


def rff_pos_encoding(input_shape):
    # pos encoding based on Random Fourier Features
    # RFF for 1D inputs (only time)
    k = input_shape[-1] // 2
    w = tf.constant(tf.random.normal([1, 1, k]))

    def pos_encoding(inputs):
        assert len(inputs.shape) == 3

        time = tf.shape(inputs)[-2]
        encoding = tf.reshape(tf.range(time, dtype=inputs.dtype), (1, -1, 1))
        encoding = tf.concat([tf.cos(w * encoding), tf.sin(w * encoding)], -1)
        return encoding
    return pos_encoding

