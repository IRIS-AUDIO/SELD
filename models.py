import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *

import layers


def seldnet(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(layers, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(layers, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(layers, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(layers, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def seldnet_v1(input_shape, model_config):
    inputs = Input(shape=input_shape[-3:])

    x = getattr(layers, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(layers, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(layers, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(layers, model_config.DOA)(model_config.DOA_ARGS)(x)

    doa *= Concatenate()([sed] * 3)
    doa = tanh(doa) 

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def seldnet_architecture(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(layers, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(layers, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(layers, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(layers, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])

def xception_gru(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(layers, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(layers, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(layers, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(layers, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])
