import copy
import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *
from utils import safe_tuple

"""
Modules

This is only for implementing modules.
Use only custom layers or predefined 
"""

"""            STAGES            """
def mother_stage(model_config: dict):
    depth = model_config['depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(x):
        for i in range(depth):
            x = mother_block(model_config)(x)
            model_config['strides'] = (1, 1)
        return x
    return stage


def simple_conv_stage(model_config: dict):
    '''
    essential configs
        filters: int
        depth: int
        pool_size: int or tuple of ints

    non-essential configs
        strides: (default=None)
        activation: (default=relu)
        dropout_rate: (default=0.)
    '''
    filters = model_config['filters']
    depth = model_config['depth']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)
    activation = model_config.get('activation', 'relu')
    strides = model_config.get('strides', None)

    def stage(x):
        for i in range(depth):
            x = conv2d_bn(filters, kernel_size=3, 
                          activation=activation)(x)
        x = MaxPooling2D(pool_size=pool_size, strides=strides)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x
    return stage


def another_conv_stage(model_config: dict):
    '''
    essential configs
        filters: int
        depth: int
        pool_size: int or tuple of ints

    non-essential configs
        activation: (default=relu)
    '''
    return another_conv_block(model_config)


def res_basic_stage(model_config: dict):
    '''
    essential configs
        depth: int
        strides: int or tuple of ints

    non-essential configs
        groups: (default=1)
        activation: (default=relu)
    '''
    depth = model_config['depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(x):
        for i in range(depth):
            x = res_basic_block(model_config)(x)
            model_config['strides'] = 1
        return x
    return stage


def res_bottleneck_stage(model_config: dict):
    '''
    essential configs
        depth: int
        strides: int or tuple of ints

    non-essential configs
        groups: (default=1)
        activation: (default=relu)
    '''
    depth = model_config['depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(x):
        for i in range(depth):
            x = res_bottleneck_block(model_config)(x)
            model_config['strides'] = 1
        return x
    return stage


def dense_net_stage(model_config: dict):
    '''
    essential configs
        growth_rate: int
        depth: int
        strides: int or tuple of ints

    non-essential configs
        bottleneck_ratio: int or float
        reduction_ratio: int or float
    '''
    return dense_net_block(model_config)


def sepformer_stage(model_config: dict):
    '''
    essential configs
        depth: int
        n_head: int
        key_dim: int
        ff_multiplier: int or float
        kernel_size: int

    non-essential configs
        pos_encoding: (default=None) [None, 'basic', 'rff']
        activation: (default=relu)
        dropout_rate: (default=0.1)
    '''
    depth = model_config['depth']

    def stage(x):
        for i in range(depth):
            x = sepformer_block(model_config)(x)
        return x
    return stage


def xception_basic_stage(model_config: dict):
    '''
    essential configs
        depth: int
        filters: int

    non-essential configs
        mid_ratio: (default=1)
        strides: (default=(1, 2))
        kernel_regularizer: (default={'l1': 0., 'l2': 0.})
    '''
    depth = model_config['depth']
    filters = model_config['filters']
    
    mid_ratio = model_config.get('mid_ratio', 1)
    strides = model_config.get('strides', (1, 2))
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    mid_filters = int(mid_ratio * filters)

    def stage(x):
        for i in range(depth):
            residual = x
            x = SeparableConv2D(mid_filters, 3, padding='same', use_bias=False, 
                                kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = SeparableConv2D(filters, 3, padding='same', use_bias=False, 
                                kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization()(x)

            if i == depth-1:
                residual = conv2d_bn(filters, 1, strides=strides, padding='same', 
                                     use_bias=False, 
                                     kernel_regularizer=kernel_regularizer, 
                                     activation=None)(residual)

                x = MaxPooling2D((3, 3), strides=strides, padding='same')(x)
            elif x.shape[-1] != residual.shape[-1]:
                residual = conv2d_bn(filters, 1, use_bias=False, 
                                     kernel_regularizer=kernel_regularizer, 
                                     activation=None)(residual)
            x = x + residual
        return x
    return stage


def bidirectional_GRU_stage(model_config: dict):
    '''
    essential configs
        depth: int
        units: int

    non-essential configs
        dropout_rate: (default=0.)
    '''
    depth = model_config['depth']
    units = model_config['units']
    model_config = copy.deepcopy(model_config)
    model_config['units'] = [units] * depth

    return bidirectional_GRU_block(model_config)


def simple_dense_stage(model_config: dict):
    '''
    essential configs
        depth: int
        units: int

    non-essential configs
        activation: (default=None)
        dropout_rate: (default=0.)
        kernel_regularizer: (default={'l1': 0., 'l2': 0.})
    '''
    depth = model_config['depth']
    units = model_config['units']
    model_config = copy.deepcopy(model_config)
    model_config['units'] = [units] * depth
    model_config['dense_activation'] = model_config.get('activation', None)
    
    return simple_dense_block(model_config)


def transformer_encoder_stage(model_config: dict):
    '''
    essential configs
        depth: int
        n_head: int
        key_dim: int
        ff_multiplier: int or float
        kernel_size: int

    non-essential configs
        activation: (default=relu)
        dropout_rate: (default=0.1)
    '''
    depth = model_config['depth']

    def stage(x):
        x = force_1d_inputs()(x)
        for i in range(depth):
            x = transformer_encoder_block(model_config)(x)
        return x
    return stage


def conformer_encoder_stage(model_config: dict):
    '''
    essential configs
        depth: int

    non-essential configs
        key_dim: (default=36)
        n_head: (default=4)
        kernel_size: (default=32)
        activation: (default=swish)
        dropout_rate: (default=0.1)
        multiplier: (default=4)
        ffn_factor: (default=0.5)
        pos_encoding: (default=basic)
        kernel_regularizer: (default={'l1': 0., 'l2': 0.})
    '''
    depth = model_config['depth']

    def stage(x):
        inputs = force_1d_inputs()(x)
        for i in range(depth):
            x = conformer_encoder_block(model_config)(x)
        return x
    return stage


"""            BLOCKS WITH 2D OUTPUTS            """
def mother_block(model_config: dict):
    filters0 = model_config['filters0'] # 0 if skipped
    filters1 = model_config['filters1'] # 0 if skipped
    filters2 = model_config['filters2'] # 0 if skipped
    kernel_size0 = model_config['kernel_size0'] # 0 if skipped
    kernel_size1 = model_config['kernel_size1'] # 0 if skipped
    kernel_size2 = model_config['kernel_size2'] # 0 if skipped
    connect0 = model_config['connect0'] # len of 1 (0: input)
    connect1 = model_config['connect1'] # len of 2 (0: input, 1: out0)
    connect2 = model_config['connect2'] # len of 3 (0: input, 1: out0, 2: out1)

    strides = safe_tuple(model_config.get('strides', (1, 1)))
    activation = model_config.get('activation', 'relu')

    if (filters0 == 0) != (kernel_size0 == 0):
        raise ValueError('0) skipped layer must have 0 filters, 0 kernel size')
    if (filters1 == 0) != (kernel_size1 == 0):
        raise ValueError('1) skipped layer must have 0 filters, 0 kernel size')
    if (filters2 == 0) != (kernel_size2 == 0):
        raise ValueError('2) skipped layer must have 0 filters, 0 kernel size')

    if filters0 == 0 and max(connect1[1], connect2[1]):
        raise ValueError('cannot link skipped layer (first layer)')
    if filters1 == 0 and connect2[2] > 0:
        raise ValueError('cannot link skipped layer (second layer)')

    if (filters0 != 0) + sum(connect0) == 0:
        raise ValueError('cannot pass zero inputs to the second layer')
    if (filters1 != 0) + sum(connect1) == 0:
        raise ValueError('cannot pass zero inputs to the third layer')
    if (filters2 != 0) + sum(connect2) == 0:
        raise ValueError('cannot pass zero inputs to the final output')

    if filters1 == 0 and tuple(strides) != (1, 1):
        raise ValueError('if strides are set, the second layer must be active')

    def block(inputs):
        outputs = [inputs]

        # first layer
        if filters0 > 0:
            out = Conv2D(filters0, kernel_size0, padding='same')(outputs[-1])
            out = BatchNormalization()(out)
            if connect0[0] == 1:
                skip = outputs[-1]
                if skip.shape[-3:] != out.shape[-3:]:
                    skip = Conv2D(filters0, 1)(skip)
                out += BatchNormalization()(skip)
            out = Activation(activation)(out)
        else:
            out = outputs[-1]
        outputs.append(out)

        # second layer (apply strides)
        if filters1 > 0:
            out = Conv2D(filters1, kernel_size1, padding='same',
                         strides=strides)(outputs[-1])
            out = BatchNormalization()(out)
            for i in range(len(connect1)):
                if connect1[i] == 1:
                    skip = outputs[i]
                    if skip.shape[-3:] != out.shape[-3:]:
                        skip = Conv2D(filters1, 1, strides=strides)(skip)
                    out += BatchNormalization()(skip)
            out = Activation(activation)(out)
        else:
            out = []
            for i in range(len(connect1)):
                if connect1[i] == 1:
                    out.append(outputs[i])
            out = tf.concat(out, axis=-1)
        outputs.append(out)

        # third layer
        if filters2 > 0:
            out = Conv2D(filters2, kernel_size2, padding='same')(outputs[-1])
            out = BatchNormalization()(out)
            for i in range(len(connect2)):
                if connect2[i] == 1:
                    skip = outputs[i]
                    if skip.shape[-3:] != out.shape[-3:]:
                        skip = Conv2D(
                            filters2, 1, 
                            strides=(1, 1) if i == 2 else strides)(skip)
                    out += BatchNormalization()(skip)
            out = Activation(activation)(out)
        else:
            out = []
            for i in range(len(connect2)):
                if connect2[i] == 1:
                    skip = outputs[i]
                    if connect2[-1] == 1 and tuple(strides) != (1, 1) and i < 2:
                        # connect with strided outputs
                        skip = Conv2D(skip.shape[-1], 1, strides=strides)(skip)
                    out.append(skip)
            out = tf.concat(out, axis=-1)

        return out
    return block


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

            out = BatchNormalization()(out)
            out = Activation(activation)(out)
            out = Conv2D(filters, 3, padding='same')(out)

            if x.shape[-1] != filters:
                x = Conv2D(filters, 1)(x)
            x = x + out

        if pool_size[0] > 1 or pool_size[1] > 1:
            x = MaxPool2D(pool_size, strides=pool_size)(x)

        return x
    return conv_block


def res_basic_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    strides = safe_tuple(model_config['strides'])

    groups = model_config.get('groups', 1)
    activation = model_config.get('activation', 'relu')

    def basic_block(inputs):
        out = Conv2D(filters, 3, strides, padding='same', groups=groups)(inputs)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

        out = Conv2D(filters, 3, padding='same', groups=groups)(out)
        out = BatchNormalization()(out)

        if strides not in [(1, 1), [1, 1]] or inputs.shape[-1] != filters:
            inputs = Conv2D(filters, 1, strides)(inputs)
            inputs = BatchNormalization()(inputs)

        return Activation(activation)(out + inputs)
    return basic_block


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

    def _dense_net_block(inputs):
        x = inputs

        for i in range(depth):
            out = BatchNormalization()(x)
            out = Activation('relu')(out)
            out = Conv2D(int(bottleneck_ratio*growth_rate), 1, 
                         use_bias=False)(out)
            out = BatchNormalization()(out)
            out = Activation('relu')(out)
            out = Conv2D(growth_rate, 3, padding='same', use_bias=True)(out)
            x = Concatenate(axis=-1)([x, out])

        if strides not in [1, (1, 1), [1, 1]]:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(int(x.shape[-1] * reduction_ratio), 1, use_bias=False)(x)
            x = AveragePooling2D(strides, strides)(x)

        return x

    return _dense_net_block


def sepformer_block(model_config: dict):
    # mandatory parameters (for transformer_encoder_block)
    # 'n_head', 'ff_multiplier', 'kernel_size'

    pos_encoding = model_config.get('pos_encoding', None)
    if pos_encoding == 'basic':
        pos_encoding = basic_pos_encoding
    elif pos_encoding == 'rff': # random fourier feature
        pos_encoding = rff_pos_encoding

    def _sepformer_block(inputs):
        # https://github.com/speechbrain/speechbrain/blob/develop/
        # speechbrain/lobes/models/dual_path.py
        # 
        # treat each chan as chunk in Sepformer
        # [batch, time, freq, chan]
        assert len(inputs.shape) == 4
        x = inputs

        batch, time, freq, chan = x.shape

        intra = tf.transpose(x, [0, 3, 1, 2])
        intra = tf.reshape(intra, [-1, time, freq])
        if pos_encoding:
            intra += pos_encoding(intra.shape)(intra)
        intra = transformer_encoder_block(model_config)(intra)
        intra = tf.reshape(intra, [-1, chan, time, freq])
        intra = tf.transpose(intra, [0, 2, 3, 1])
        intra = LayerNormalization()(intra) + x

        inter = tf.transpose(x, [0, 1, 3, 2]) 
        inter = tf.reshape(inter, [-1, chan, freq])
        if pos_encoding:
            inter += pos_encoding(inter.shape)(inter)
        inter = transformer_encoder_block(model_config)(inter)
        inter = tf.reshape(inter, [-1, time, chan, freq])
        inter = tf.transpose(inter, [0, 1, 3, 2])
        inter = LayerNormalization()(inter) + intra

        return inter

    return _sepformer_block
    

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
        return x
    return _xception_net_block


def xception_basic_block(model_config: dict):
    filters = model_config['filters']
    
    mid_ratio = model_config.get('mid_ratio', 1)
    strides = model_config.get('strides', (1, 2))
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    mid_filters = int(mid_ratio * filters)

    def _basic_block(inputs):
        x = SeparableConv2D(mid_filters, 3, padding='same', use_bias=False, 
                            kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(filters, 3, padding='same', use_bias=False, 
                            kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=strides, padding='same')(x)

        residual = conv2d_bn(filters, 1, strides=strides, padding='same', 
                             use_bias=False, 
                             kernel_regularizer=kernel_regularizer, 
                             activation=None)(inputs)
        return x + residual
    return _basic_block


"""            BLOCKS WITH 1D OUTPUTS            """
def bidirectional_GRU_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']

    dropout_rate = model_config.get('dropout_rate', 0.)

    def GRU_block(inputs):
        x = force_1d_inputs()(inputs)

        for units in units_per_layer:
            x = Bidirectional(
                GRU(units, activation='tanh', 
                    dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)
        return x

    return GRU_block


def simple_dense_block(model_config: dict):
    # assumes 1D inputs
    # mandatory parameters
    units_per_layer = model_config['units']

    activation = model_config.get('dense_activation', None)
    dropout_rate = model_config.get('dropout_rate', 0)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def dense_block(inputs):
        x = force_1d_inputs()(inputs)

        for units in units_per_layer:
            if len(x.shape) == 2:
                x = Dense(units, activation=activation,
                          kernel_regularizer=kernel_regularizer)(x)
            else:
                x = TimeDistributed(
                    Dense(units, activation=activation,
                          kernel_regularizer=kernel_regularizer))(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        return x

    return dense_block


def transformer_encoder_block(model_config: dict):
    n_head = model_config['n_head']
    key_dim = model_config['key_dim']
    ff_multiplier = model_config['ff_multiplier'] # default to 4 
    kernel_size = model_config['kernel_size'] # default to 1

    activation = model_config.get('activation', 'relu')
    dropout_rate = model_config.get('dropout_rate', 0.1)
    
    def block(inputs):
        x = force_1d_inputs()(inputs)
        d_model = x.shape[-1]

        attn = MultiHeadAttention(
            n_head, key_dim, dropout=dropout_rate)(x, x)
        attn = Dropout(dropout_rate)(attn)
        x = LayerNormalization()(x + attn)

        # FFN
        ffn = Conv1D(int(ff_multiplier*d_model), kernel_size, padding='same',
                     activation=activation)(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Conv1D(d_model, kernel_size, padding='same')(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = LayerNormalization()(x + ffn)

        return x

    return block


def conformer_encoder_block(model_config: dict):
    # mandatory parameters
    key_dim = model_config.get('key_dim', 36)
    n_head = model_config.get('n_head', 4)
    kernel_size = model_config.get('kernel_size', 32) # 32 
    activation = model_config.get('activation', 'swish')
    dropout_rate = model_config.get('dropout_rate', 0.1)
    multiplier = model_config.get('multiplier', 4)
    ffn_factor = model_config.get('ffn_factor', 0.5)
    pos_encoding = model_config.get('pos_encoding', 'basic')
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if pos_encoding == 'basic':
        pos_encoding = basic_pos_encoding
    elif pos_encoding == 'rff': # random fourier feature
        pos_encoding = rff_pos_encoding
    else:
        pos_encoding = None
    
    def conformer_block(inputs):
        inputs = force_1d_inputs()(inputs)
        x = inputs
        batch, time, emb = x.shape

        # FFN Modules
        ffn = LayerNormalization()(x)
        ffn = Dense(multiplier*emb, activation=activation)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(emb)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = x + ffn_factor*ffn

        # Positional Encoding
        if pos_encoding:
            x += pos_encoding(x.shape)(x)
            
        # Multi Head Self Attention module
        attn = LayerNormalization()(x)
        attn = MultiHeadAttention(n_head,
                                  key_dim,
                                  dropout=dropout_rate)(attn, attn)
        attn = Dropout(dropout_rate)(attn)
        x = attn + x

        # Conv Module
        conv = LayerNormalization()(x)
        conv = Conv1D(filters=2*emb, 
                      kernel_size=1,
                      kernel_regularizer=kernel_regularizer)(conv)
        
        # GLU Part
        conv_1, conv_2 = tf.split(conv, 2, axis=-1)
        conv_1 = Dense(emb)(conv_1)
        conv_2 = Dense(emb, activation ='sigmoid')(conv_2)
        conv = conv_1 * conv_2

        #Depth Wise
        conv = Conv1D(filters=emb,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      groups=emb, 
                      kernel_regularizer=kernel_regularizer)(conv)

        conv = BatchNormalization()(conv)
        conv = tf.keras.activations.swish(conv) 
        conv = Conv1D(filters=emb, 
                      kernel_size=1,
                      padding="same",
                      kernel_regularizer=kernel_regularizer)(conv)
        conv = Dropout(dropout_rate)(conv)
        conv = conv + x

        # FFN
        ffn = LayerNormalization()(conv)
        ffn = Dense(multiplier*emb, activation=activation)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(emb)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        
        x = LayerNormalization()(x + ffn_factor*ffn)

        return x

    return conformer_block


"""                 OTHER BLOCKS                 """
def identity_block(model_config: dict):
    def identity(inputs):
        return inputs
    return identity

