import numpy as np
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
    n_classes = model_config.get('n_classes', 14)

    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config['FIRST'])(model_config['FIRST_ARGS'])(inputs)
    x = getattr(modules, model_config['SECOND'])(model_config['SECOND_ARGS'])(x)

    sed = getattr(modules, model_config['SED'])(model_config['SED_ARGS'])(x)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)
    doa = getattr(modules, model_config['DOA'])(model_config['DOA_ARGS'])(x)
    doa = Dense(3*n_classes, activation='tanh', name='doa_out')(doa)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def seldnet_v1(input_shape, model_config):
    n_classes = model_config.get('n_classes', 14)

    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config['FIRST'])(model_config['FIRST_ARGS'])(inputs)
    x = getattr(modules, model_config['SECOND'])(model_config['SECOND_ARGS'])(x)

    sed = getattr(modules, model_config['SED'])(model_config['SED_ARGS'])(x)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)
    doa = getattr(modules, model_config['DOA'])(model_config['DOA_ARGS'])(x)
    doa = Dense(3*n_classes, activation='tanh', name='doa_out')(doa)

    doa *= Concatenate()([sed] * 3)
    doa = tanh(doa) 

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])
    

def conv_temporal(input_shape, model_config):
    filters = model_config.get('filters', 32)
    first_kernel_size = model_config.get('first_kernel_size', 7)
    first_pool_size = model_config.get('first_pool_size', [5, 1])
    n_classes = model_config.get('n_classes', 14)

    inputs = Input(shape=input_shape[-3:])
    
    x = layers.conv2d_bn(filters, first_kernel_size, padding='same', 
                         activation='relu')(inputs)
    x = MaxPooling2D(first_pool_size, padding='same')(x)

    blocks = [key for key in model_config.keys()
              if key.startswith('BLOCK') and not key.endswith('_ARGS')]
    blocks.sort()

    for block in blocks:
        x = getattr(modules, model_config[block])(model_config[f'{block}_ARGS'])(x)

    sed = getattr(modules, model_config['SED'])(model_config['SED_ARGS'])(x)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)
    doa = getattr(modules, model_config['DOA'])(model_config['DOA_ARGS'])(x)
    doa = Dense(3*n_classes, activation='tanh', name='doa_out')(doa)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def vad_architecture(input_shape, model_config):
    flatten = model_config.get('flatten', True)
    last_unit = model_config.get('last_unit', 1)

    inputs = Input(shape=input_shape)

    x = inputs
    if flatten:
        x = Reshape((np.prod(input_shape),))(x)

    blocks = sorted([key for key in model_config.keys()
                     if key.startswith('BLOCK') and not key.endswith('_ARGS')])

    for block in blocks:
        x = getattr(modules, model_config[block])(
            model_config[f'{block}_ARGS'])(x)

    x = Dense(last_unit, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

