import copy
import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *

"""
Modules

This is only for implementing modules.
Use only custom layers or predefined 
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
            x = conv2d_bn(filters[i], kernel_size=3, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x
    return conv_block


def another_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    depth = model_config['depth']
    pool_size = model_config['pool_size']

    activation = model_config.get('activation', 'relu')

    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)

    def conv_block(inputs):
        x = inputs

        for i in range(depth):
            out = BatchNormalization()(x)
            out = Activation(activation)(out)
            out = Conv2D(filters, 3, padding='same')(out)

            out = BatchNormalization()(x)
            out = Activation(activation)(out)
            out = Conv2D(filters, 3, padding='same')(out)

            if x.shape[-1] != filters:
                x = Conv2D(filters, 1)(x)
            x = x + out

        if pool_size[0] > 1 or pool_size[1] > 1:
            x = MaxPool2D(pool_size, strides=pool_size)(x)

        return x
    return conv_block


def res_basic_stage(model_config: dict):
    # mandatory parameters
    depth = model_config['depth']
    strides = model_config['strides']

    model_config = copy.deepcopy(model_config)

    def stage(inputs):
        x = inputs
        for i in range(depth):
            x = res_basic_block(model_config)(x)
            model_config['strides'] = 1
        return x
    return stage


def res_basic_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    strides = model_config['strides']

    groups = model_config.get('groups', 1)
    activation = model_config.get('activation', 'relu')

    if isinstance(strides, int):
        strides = (strides, strides)

    def basic_block(inputs):
        out = Conv2D(filters, 3, strides, padding='same', groups=groups)(inputs)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv2D(filters, 3, padding='same', groups=groups)(out)
        out = BatchNormalization()(out)

        if strides not in [(1, 1), [1, 1]] or inputs.shape[-1] != filters:
            inputs = Conv2D(filters, 1, strides)(inputs)
            inputs = BatchNormalization()(inputs)

        out = Activation(activation)(out + inputs)

        return out

    return basic_block


def res_bottleneck_stage(model_config: dict):
    # mandatory parameters
    depth = model_config['depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(inputs):
        x = inputs
        for i in range(depth):
            x = res_bottleneck_block(model_config)(x)
            model_config['strides'] = 1
        return x
    return stage


def res_bottleneck_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    strides = model_config['strides']

    groups = model_config.get('groups', 1)
    bottleneck_ratio = model_config.get('bottleneck_ratio', 1)
    activation = model_config.get('activation', 'relu')

    if isinstance(strides, int):
        strides = (strides, strides)
    bottleneck_size = int(filters * bottleneck_ratio)

    def bottleneck_block(inputs):
        out = Conv2D(bottleneck_size, 1)(inputs)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv2D(bottleneck_size, 3, strides, 
                     padding='same', groups=groups)(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv2D(filters, 1)(out)
        out = BatchNormalization()(out)

        if strides not in [(1, 1), [1, 1]] or inputs.shape[-1] != filters:
            inputs = Conv2D(filters, 1, strides)(inputs)
            inputs = BatchNormalization()(inputs)

        out = Activation(activation)(out + inputs)

        return out

    return bottleneck_block


def dense_net_block(model_config: dict):
    # mandatory
    growth_rate = model_config['growth_rate']
    depth = model_config['depth']
    strides = model_config['strides']

    bottleneck_ratio = model_config.get('bottleneck_ratio', 4)
    reduction_ratio = model_config.get('reduction_ratio', 0.5)
    bn_args = model_config.get('bn_args', {})

    def _dense_net_block(inputs):
        x = inputs

        for i in range(depth):
            out = BatchNormalization(**bn_args)(x)
            out = Activation('relu')(out)
            out = Conv2D(bottleneck_ratio * growth_rate, 1, use_bias=False)(out)
            out = BatchNormalization(**bn_args)(out)
            out = Activation('relu')(out)
            out = Conv2D(growth_rate, 3, padding='same', use_bias=True)(out)
            x = Concatenate(axis=-1)([x, out])

        if strides not in [1, (1, 1), [1, 1]]:
            x = BatchNormalization(**bn_args)(x)
            x = Activation('relu')(x)
            x = Conv2D(int(x.shape[-1] * reduction_ratio), 1, use_bias=False)(x)
            x = AveragePooling2D(strides, strides)(x)

        return x

    return _dense_net_block


def xception_block(model_config: dict):
    filters = model_config['filters']
    block_num = model_config['block_num']
    
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def _sepconv_block(inputs, filters, activation):
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False, 
                            kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x) if activation else x
        return x
    
    def _residual_block(inputs, filters):
        if type(filters) != list:
            filters1 = filters2 = filters
        else:
            filters1, filters2 = filters

        residual = conv2d_bn(filters2, 1, strides=(1,2), padding='same', 
                             use_bias=False, 
                             kernel_regularizer=kernel_regularizer, 
                             activation=None)(inputs)

        x = _sepconv_block(inputs, filters1, 'relu')
        x = _sepconv_block(x, filters2, None)
        x = MaxPooling2D((3,3), strides=(1,2), padding='same')(x)

        x = add([x, residual])
        return x

    def _xception_net_block(inputs):
        x = conv2d_bn(filters, 3, use_bias=False, 
                      kernel_regularizer=kernel_regularizer)(inputs)
        x = MaxPooling2D(pool_size=(5,1))(x)
        x = conv2d_bn(filters * 2, 3, use_bias=False, 
                      kernel_regularizer=kernel_regularizer)(x)

        x = _residual_block(x, filters * 4)
        x = _residual_block(x, filters * 8)
        x = _residual_block(x, int(filters * 22.75))

        for i in range(block_num):
            residual = x

            x = Activation('relu')(x)
            x = _sepconv_block(x, int(filters * 22.75), 'relu')
            x = _sepconv_block(x, int(filters * 22.75), 'relu')
            x = _sepconv_block(x, int(filters * 22.75), None)

            x = add([x, residual])

        x = _residual_block(x, [int(filters * 22.75), filters * 32])

        x = _sepconv_block(x, filters * 48, 'relu')
        x = _sepconv_block(x, filters * 64, 'relu')
        x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)
        return x
    return _xception_net_block
    

"""      sequential blocks      """
def bidirectional_GRU_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']

    dropout_rate = model_config.get('dropout_rate', 0.)

    def GRU_block(inputs):
        x = inputs
        x_shape = x.shape
        if len(x_shape) == 4: # [batch, time, freq, chan]
            x = Reshape((-1, x_shape[-2]*x_shape[-1]))(x)

        for units in units_per_layer:
            x = Bidirectional(
                GRU(units, activation='tanh', 
                    dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)

        if len(x_shape) == 4:
            x = Reshape((*x_shape[-3:-1], -1))(x)

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


def identity_block(model_config: dict):
    def identity(inputs):
        return inputs
    return identity

