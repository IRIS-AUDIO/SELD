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

    x = layers.force_1d_inputs()(x)
    x = Dense(last_unit, activation='sigmoid')(x)
    if x.shape[-1] == 1:
        x = x[..., 0]
    return tf.keras.Model(inputs=inputs, outputs=x)


def spectro_temporal_attention_based_VAD(input_shape, model_config):
    T = model_config.get('T', 4) # depth of spectral attention stage
    Nc = model_config.get('Nc', 16) # filters in spectral attention stage
    fc = model_config.get('fc', 3) # kernel_size in spectral attention stage
    Np = model_config.get('Np', 256) # units in pipe-net
    Nt = model_config.get('Nt', 128) # units in temporal attention stage
    H = model_config.get('H', 4) # n_heads in temporal attention stage

    dropout_rate = model_config.get('dropout_rate', 0.5)

    inputs = Input(shape=input_shape)
    x = inputs

    # spectral attention
    for i in range(T):
        x = layers.conv2d_bn(Nc*(2**i), fc, activation=None)(x) \
          * layers.conv2d_bn(Nc*(2**i), fc, activation='sigmoid')(x)
        x = MaxPooling2D(pool_size=[1, 2])(x)
    x = Reshape((x.shape[-3], -1))(x)

    # pipe net
    for i in range(2):
        x = Dense(Np)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
    pipe = Dense(1, activation='sigmoid')(x)

    # temporal attention
    # q: [batch, Nt], k: [batch, time, Nt], v: [batch, time, Nt]
    query = Dense(Nt, use_bias=False)(tf.reduce_mean(x, axis=-2))
    query = BatchNormalization()(query)
    query = Activation('sigmoid')(query)
    key = Dense(Nt, use_bias=False)(x)
    key = BatchNormalization()(key)
    key = Activation('sigmoid')(key)
    value = Dense(Nt, use_bias=False)(x)
    value = BatchNormalization()(value)
    value = Activation('sigmoid')(value)

    scale = 1 / tf.sqrt(tf.cast(Nt, dtype=tf.float32))
    query = Reshape((*query.shape[1:-1], Nt//H, H))(query)
    key = Reshape((*key.shape[1:-1], Nt//H, H))(key)
    value = Reshape((*value.shape[1:-1], Nt//H, H))(value)
    
    score = tf.reduce_sum(query[:, tf.newaxis, ...] * key, axis=-2) * scale
    x = value * tf.nn.softmax(score[..., tf.newaxis, :], axis=-3)
    x = Reshape((*x.shape[1:-2], Nt))(x)
    score = tf.nn.softmax(tf.reduce_sum(score, axis=-1), axis=-1)

    # post net
    for i in range(1):
        x = Dense(Np)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=[x, pipe, score])

