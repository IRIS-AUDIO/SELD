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


def xception_block(model_config: dict):
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)

    padding = 'same'
    if type(filters) != int:
        if len(filters) == 0:
            filters = filters * len(pool_size)
        elif len(filters) != len(pool_size):
            raise ValueError("len of filters and pool_size do not match")

    def _xception_block(inputs):
        x = TimeDistributed(Conv1D(
            filters, 3,
            strides=2,
            use_bias=False,
            name='block1_conv1'))(inputs)
        x = MaxPooling2D((5, 1))(x)
        x = TimeDistributed(BatchNormalization(name='block1_conv1_bn'))(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = TimeDistributed(Conv1D(x.shape[-1] * 2, 3, use_bias=False, name='block1_conv2'))(x)
        x = TimeDistributed(BatchNormalization(name='block1_conv2_bn'))(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = TimeDistributed(Conv1D(
            x.shape[-1] * 2, 1, strides=2, padding='same', use_bias=False))(x)
        residual = TimeDistributed(BatchNormalization())(residual)

        x = TimeDistributed(SeparableConv1D(
            128, 3, padding='same', use_bias=False, name='block2_sepconv1'))(x)
        x = TimeDistributed(BatchNormalization(name='block2_sepconv1_bn'))(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = TimeDistributed(SeparableConv1D(
            128, 3, padding='same', use_bias=False, name='block2_sepconv2'))(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = TimeDistributed(MaxPooling1D(3,
                        strides=2,
                        padding='same',
                        name='block2_pool'))(x)
        x = add([x, residual])

        residual = TimeDistributed(Conv1D(
            256, 1, strides=2, padding='same', use_bias=False))(x)
        residual = TimeDistributed(BatchNormalization())(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = TimeDistributed(SeparableConv1D(
            256, 3, padding='same', use_bias=False, name='block3_sepconv1'))(x)
        x = TimeDistributed(BatchNormalization(name='block3_sepconv1_bn'))(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = TimeDistributed(SeparableConv1D(
            256, 3, padding='same', use_bias=False, name='block3_sepconv2'))(x)
        x = TimeDistributed(BatchNormalization(name='block3_sepconv2_bn'))(x)

        x = TimeDistributed(MaxPooling1D(3,
                        strides=2,
                        padding='same',
                        name='block3_pool'))(x)
        x = add([x, residual])

        residual = TimeDistributed(Conv1D(
            728, 1, strides=2, padding='same', use_bias=False))(x)
        residual = TimeDistributed(BatchNormalization())(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = TimeDistributed(SeparableConv1D(
            728, 3, padding='same', use_bias=False, name='block4_sepconv1'))(x)
        x = TimeDistributed(BatchNormalization(name='block4_sepconv1_bn'))(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = TimeDistributed(SeparableConv1D(
            728, 3, padding='same', use_bias=False, name='block4_sepconv2'))(x)
        x = TimeDistributed(BatchNormalization(name='block4_sepconv2_bn'))(x)

        x = TimeDistributed(MaxPooling1D(3,
                                strides=2,
                                padding='same',
                                name='block4_pool'))(x)
        x = add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = TimeDistributed(SeparableConv1D(
                728, 3,
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv1'))(x)
            x = TimeDistributed(BatchNormalization(
                name=prefix + '_sepconv1_bn'))(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = TimeDistributed(SeparableConv1D(
                728, 3,
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv2'))(x)
            x = TimeDistributed(BatchNormalization(
                name=prefix + '_sepconv2_bn'))(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = TimeDistributed(SeparableConv1D(
                728, 3,
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv3'))(x)
            x = BatchNormalization(
                name=prefix + '_sepconv3_bn')(x)

            x = add([x, residual])

        return x
    return _xception_block