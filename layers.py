import tensorflow as tf
from tensorflow.keras.layers import *

"""
Layers

This is only for implementing layers.
You should not import class or functions from modules or models
"""


def conv2d_layer(filters,
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

