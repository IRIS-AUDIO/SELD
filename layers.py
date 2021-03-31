import tensorflow as tf
from tensorflow.keras.layers import *

"""
Layers

This is only for implementing layers.
You should not import class or functions from modules or models
"""


def conv2d_block(filters,
                 kernel_size, 
                 strides=(1, 1), 
                 padding='same', 
                 activation='relu', 
                 use_bias=True, 
                 kernel_regularizer=None, 
                 groups=1,
                 norm_axis=-1,
                 norm_eps=1e-3):
    def _conv2d_block(inputs):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_regularizer=kernel_regularizer, groups=groups)(inputs)
        x = BatchNormalization(norm_axis, epsilon=norm_eps)(x)
        x = Activation(activation)(x) if activation else x
        return x
    return _conv2d_block

