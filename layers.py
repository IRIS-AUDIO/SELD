import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *


def high_level_feature_representation_base(config):
    t_pool_size = [int(i) for i in config.high_level_feature_representation_pool1.split(',')]
    pool_size = [int(i) for i in config.high_level_feature_representation_pool2.split(',')]
    
    def _high_level_feature_representation(inputs):
        x = inputs
        for i, convCnt in enumerate(pool_size):
            x = Conv2D(config.high_level_feature_representation_filter, kernel_size=3, padding='same')(
                x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(t_pool_size[i], pool_size[i]))(
                x)
            x = Dropout(config.dropout_rate)(x)
        outputs = x
        return outputs
    return _high_level_feature_representation


def temporal_context_representation_base(config):
    rnn_size = [int(i) for i in config.temporal_context_representation_filter.split(',')]

    def _temporal_context_representation(inputs):
        x = inputs
        # [b, t, f, c] -> [b, t, c]
        x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)
        for nb_rnn_filt in rnn_size:
            x = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', 
                    dropout=config.dropout_rate, recurrent_dropout=config.dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)
        outputs = x
        return outputs
    return _temporal_context_representation


def sed_layer_base(config, n_classes):
    fnn_size = [int(i) for i in config.sed_layer_filter.split(',')]
    
    def _sed_layer(inputs):
        x = inputs
        for nb_fnn_filt in fnn_size:
            x = TimeDistributed(Dense(nb_fnn_filt))(x)
            x = Dropout(config.dropout_rate)(x)
        x = TimeDistributed(Dense(n_classes, name='sed_out'))(x)
        outputs = x
        return outputs
    return _sed_layer


def doa_layer_base(config, n_classes):
    fnn_size = [int(i) for i in config.doa_layer_filter.split(',')]

    def _doa_layer(inputs):
        x = inputs
        for nb_fnn_filt in fnn_size:
            x = TimeDistributed(Dense(nb_fnn_filt))(x)
            x = Dropout(config.dropout_rate)(x)
        x = TimeDistributed(Dense(n_classes*3, name='doa_out'))(x) 
        outputs = x
        return outputs
    return _doa_layer


def simple_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = Conv2D(filters[i], kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


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


def simple_dense_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']
    n_classes = model_config['n_classes']

    dropout_rate = model_config.get('dropout_rate', 0)
    activation = model_config.get('activation', None)
    name = model_config.get('name', None)

    def dense_block(inputs):
        x = inputs
        for units in units_per_layer:
            x = TimeDistributed(Dense(units))(x)
            x = Dropout(dropout_rate)(x)
        x = TimeDistributed(
            Dense(n_classes, activation=activation, name=name))(x) 
        return x

    return dense_block
