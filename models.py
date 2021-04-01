import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *

import modules
import layers


"""
Models

This defines model structures (architectures)
You should not define modules nor layers here.
"""


def seldnet(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def seldnet_v1(input_shape, model_config):
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    doa *= Concatenate()([sed] * 3)
    doa = tanh(doa) 

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])
    

def conv_temporal(input_shape, model_config):
    inputs = Input(shape=input_shape[-3:])

    filters = 24
    pool_size = [5, 2]
    
    x = layers.conv2d_bn(filters, 7, padding='same', activation='relu')(inputs)
    if pool_size[0] > 1 or pool_size[1] > 1:
        x = MaxPooling2D(pool_size, padding='same')(x)

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(x)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    x = getattr(modules, model_config.THIRD)(model_config.THIRD_ARGS)(x)
    x = getattr(modules, model_config.FOURTH)(model_config.FOURTH_ARGS)(x)
    x = getattr(modules, model_config.FIFTH)(model_config.FIFTH_ARGS)(x)

    x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])

