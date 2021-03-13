import copy
import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *

"""
Modules

This is only for implementing modules.
Use only custom layers or predefined layers.
"""

"""      conv based blocks      """
def simple_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = Conv2D(filters[i], kernel_size=3, padding='same', 
                       kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def dynamic_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']
    
    dropout_rate = model_config.get('dropout_rate', 0.)
    activation = model_config.get('activation', 'softmax')    
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = DConv2D(filters[i], kernel_size=3, padding='same', activation=activation)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def cond_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']
    
    dropout_rate = model_config.get('dropout_rate', 0.)  
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = CondConv2D(filters[i], kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def xception_block(model_config: dict):
    filters = model_config['filters']
    block_num = model_config['block_num']
    
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def _xception_block(inputs):
        x = Conv2D(filters, 3, name='block1_conv2', padding='same', 
                   kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)
        x = MaxPooling2D(pool_size=(5,1))(x)

        for _ in range(block_num):
            residual = Conv2D(x.shape[-1]*2, (1, 1), strides=(1,1), 
                              padding='same', 
                              kernel_regularizer=kernel_regularizer)(x)
            residual = BatchNormalization()(residual)

            x = SeparableConv2D(x.shape[-1]*2, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(x.shape[-1], 3, padding='same')(x)
            x = BatchNormalization()(x)

            x = add([x, residual])

        return x
    return _xception_block
    

"""      sequential blocks      """
def bidirectional_GRU_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']

    dropout_rate = model_config.get('dropout_rate', 0.)

    def GRU_block(inputs):
        x = inputs
        if len(x.shape) == 4: # [batch, time, freq, chan]
            x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

        for units in units_per_layer:
            x = Bidirectional(
                GRU(units, activation='tanh', 
                    dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)
        return x

    return GRU_block


def transformer_encoder_layer(model_config: dict):
    # mandatory parameters
    d_model = model_config['d_model']
    n_head = model_config['n_head']

    activation = model_config.get('activation', 'relu')
    dim_feedforward = model_config.get('dim_feedforward', d_model*4)
    dropout_rate = model_config.get('dropout_rate', 0.1)
    
    def block(inputs):
        assert inputs.shape[-1] == d_model
        x = inputs
        attn = MultiHeadAttention(
            n_head, d_model//n_head, dropout=dropout_rate)(x, x)
        attn = Dropout(dropout_rate)(attn)
        x = LayerNormalization()(x + attn)

        # FFN
        ffn = Dense(dim_feedforward, activation=activation)(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = LayerNormalization()(x + ffn)

        return x

    return block


"""      other blocks      """
def simple_dense_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']
    n_classes = model_config['n_classes']

    name = model_config.get('name', None)
    activation = model_config.get('activation', None)
    dropout_rate = model_config.get('dropout_rate', 0)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def dense_block(inputs):
        x = inputs
        for units in units_per_layer:
            x = TimeDistributed(
                Dense(units, kernel_regularizer=kernel_regularizer))(x)
            x = Dropout(dropout_rate)(x)
        x = TimeDistributed(
            Dense(n_classes, activation=activation, name=name,
                  kernel_regularizer=kernel_regularizer))(x) 
        return x

    return dense_block

