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
    '''
    essential configs
        filters0: int
        filters1: int
        filters2: int
        kernel_size0: int
        kernel_size1: int
        kernel_size2: int
        connect0: [int]
        connect1: [int, int]
        connect2: [int, int, int]

    non-essential configs
        strides: (default=(1, 1))
        activation: (default=relu)
        squeeze_ratio: (default=0)
        se_activation: (default=relu)
    '''
    depth = model_config['depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(x):
        for i in range(depth):
            x = mother_block(model_config)(x)
            model_config['strides'] = (1, 1)
        return x
    return stage


def bidirectional_GRU_stage(model_config: dict):
    '''
    DEPRECATED!!!
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


def RNN_stage(model_config: dict):
    '''
    essential configs
        depth: int
        units: int

    non-essential configs
        bidirectional: (default=True)
        merge_mode: (default='mul') or concat, ave
        rnn_type: (default='GRU') or LSTM
        dropout_rate: (default=0.)
    '''
    depth = model_config['depth']
    units = model_config['units']

    def stage(x):
        for i in range(depth):
            x = RNN_block(model_config)(x)
        return x
    return stage


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
        for i in range(depth):
            x = conformer_encoder_block(model_config)(x)
        return x
    return stage


def attention_stage(model_config: dict):
    '''
    essential configs
        depth: int
        key_dim: int
        n_head: int
        kernel_size: int
        ff_kernel_size: int
        ff_multiplier: float
        ff_factor0: float
        ff_factor1: float

    non-essential configs
        activation: (default=swish)
        pos_encoding: (default='basic')
        abs_pos_encoding: (default='False)
        layer_norm_in_front: (default=False)
        use_glu: (default=False)
    '''
    depth = model_config['depth']

    def stage(x):
        for i in range(depth):
            x = attention_block(model_config)(x)
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

    # squeeze and excitation
    squeeze_ratio = model_config.get('squeeze_ratio', 0)
    se_activation = model_config.get('se_activation', 'relu')

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
                    skip = BatchNormalization()(skip)
                out += skip
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
                        skip = BatchNormalization()(skip)
                    out += skip
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
                        skip = BatchNormalization()(skip)
                    out += skip
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

        # Squeeze and Excitation
        if squeeze_ratio > 0:
            se_filters = int(squeeze_ratio * out.shape[-1])

            se = tf.reduce_mean(out, axis=(-3, -2), keepdims=True)
            se = Conv2D(se_filters, 1, activation=se_activation)(se)
            se = Conv2D(out.shape[-1], 1, activation='sigmoid')(se)
            out = se * out

        return out
    return block


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


def RNN_block(model_config: dict):
    # mandatory parameters
    units = model_config['units']

    bidirectional = model_config.get('bidirectional', True)
    merge_mode = model_config.get('merge_mode', 'mul') # mul, concat, avg
    rnn_type = model_config.get('rnn_type', 'GRU')
    dropout_rate = model_config.get('dropout_rate', 0.)

    def block(inputs, rnn_type=rnn_type):
        x = force_1d_inputs()(inputs)

        if rnn_type == 'GRU':
            rnn_type = GRU
        else:
            rnn_type = LSTM
        main_block = rnn_type(
            units, dropout=dropout_rate, recurrent_dropout=dropout_rate, 
            return_sequences=True)
        if bidirectional:
            main_block = Bidirectional(main_block, merge_mode=merge_mode)

        x = main_block(x)
        return x

    return block


def simple_dense_block(model_config: dict):
    # assumes 1D inputs
    # mandatory parameters
    units_per_layer = model_config['units']

    kernel_size = model_config.get('kernel_size', 1)
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
                x = Conv1D(units, kernel_size, padding='same',
                           activation=activation,
                           kernel_regularizer=kernel_regularizer)(x)
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
    pos_mode = model_config.get('pos_mode', 'absolute')
    use_bias = model_config.get('use_bias', True)

    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))
    if pos_encoding == 'basic':
        pos_encoding_ = basic_pos_encoding
    elif pos_encoding == 'rff': # random fourier feature
        pos_encoding_ = rff_pos_encoding
    else:
        pos_encoding_ = None
    
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
        if pos_encoding in ['basic','rff']:
            encoding = pos_encoding_(x.shape)(x)

        if pos_mode == 'absolute':
            x = x + encoding

        # Multi Head Self Attention module
        attn = LayerNormalization()(x)

        if pos_mode == 'relative':
            attn = RelPositionMultiHeadAttention(n_head,
                                    key_dim,
                                    use_bias=use_bias,
                                    dropout=dropout_rate)([attn, attn, attn, encoding])
        else:
            attn = MultiHeadAttention_(n_head,
                                    key_dim,
                                    use_bias=use_bias,
                                    dropout=dropout_rate)([attn, attn, attn])

        attn = Dropout(dropout_rate)(attn)
        x = attn + x

        # Conv Module
        conv = LayerNormalization()(x)
        conv = Conv1D(filters=2*emb, 
                      kernel_size=1,
                      kernel_regularizer=kernel_regularizer)(conv)
        
        # GLU Part
        conv_1, conv_2 = tf.split(conv, 2, axis=-1)
        conv_2 = tf.keras.activations.sigmoid(conv_2)
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


def attention_block(model_config: dict):
    # mandatory parameters
    key_dim = model_config['key_dim']
    n_head = model_config['n_head']
    kernel_size = model_config['kernel_size'] # depthwise conv
    ff_kernel_size = model_config['ff_kernel_size']
    ff_multiplier = model_config['ff_multiplier']
    ff_factor0 = model_config['ff_factor0']
    ff_factor1 = model_config['ff_factor1']

    activation = model_config.get('activation', 'swish')
    pos_encoding = model_config.get('pos_encoding', 'basic')
    abs_pos_encoding = model_config.get('abs_pos_encoding', False)
    layer_norm_in_front = model_config.get('layer_norm_in_front', False)
    use_glu = model_config.get('use_glu', False)

    # do not tune these parameters unless you really need to
    use_bias = model_config.get('use_bias', False)
    dropout_rate = model_config.get('dropout_rate', 0.1)

    use_depthwise_conv = kernel_size > 0
    if pos_encoding == 'basic':
        pos_encoding = basic_pos_encoding
    elif pos_encoding == 'rff': # random fourier feature
        pos_encoding = rff_pos_encoding
    else: # is None
        pos_encoding = lambda x: (lambda x: 0)

    # raising errors
    if ff_factor0 < 0 or ff_factor1 < 0:
        raise ValueError('ff_factor0, ff_factor1 >= 0 must hold')
    if ff_factor0 == 0 and ff_factor1 == 0:
        if ff_kernel_size > 0:
            raise ValueError('if FF modules are not used, '
                             'ff_kernel must be set to 0')
        if ff_multiplier > 0:
            raise ValueError('if FF modules are not used, '
                             'ff_multiplier must be set to 0')
    if not abs_pos_encoding and pos_encoding is None:
        raise ValueError('relative pos encoding demands any types of encoding '
                         'except the null one')

    def attention_block(inputs, pos_encoding=pos_encoding):
        inputs = force_1d_inputs()(inputs)
        x = inputs
        batch, time, d_model = x.shape

        # First FF
        if ff_factor0 > 0:
            ff = x
            if layer_norm_in_front:
                ff = LayerNormalization()(ff)

            ff = Conv1D(int(ff_multiplier * d_model), ff_kernel_size,
                        padding='same', activation=activation)(x)
            ff = Dropout(dropout_rate)(ff)
            ff = Conv1D(d_model, ff_kernel_size, padding='same')(ff)
            ff = Dropout(dropout_rate)(ff)
            x = x + ff_factor0 * ff

            if not layer_norm_in_front:
                x = LayerNormalization()(x)

        # Multi Head Self Attention
        attn = x
        pos_encoding = pos_encoding(x.shape)(x)

        if layer_norm_in_front:
            attn = LayerNormalization()(attn)
        if abs_pos_encoding:
            x = x + pos_encoding
            attn = MultiHeadAttention_(
                n_head, key_dim, use_bias=use_bias,
                dropout=dropout_rate)([attn, attn, attn])
        else: # not abs_pos_encoding:
            attn = RelPositionMultiHeadAttention(
                n_head, key_dim, use_bias=use_bias,
                dropout=dropout_rate)([attn, attn, attn, pos_encoding])
        x = Dropout(dropout_rate)(attn) + x
        if not layer_norm_in_front:
            x = LayerNormalization()(x)

        # GLU
        conv = x
        if use_glu:
            if layer_norm_in_front:
                conv = LayerNormalization()(conv)
            conv = Conv1D(filters=2*d_model, kernel_size=1)(conv)
            conv_1, conv_2 = tf.split(conv, 2, axis=-1)
            conv_2 = tf.keras.activations.sigmoid(conv_2)
            conv = conv_1 * conv_2

        # Depth Wise
        if use_depthwise_conv:
            if layer_norm_in_front and not use_glu:
                conv = LayerNormalization()(conv)
            conv = Conv1D(filters=d_model, kernel_size=kernel_size,
                          strides=1, padding='same', groups=d_model)(conv)
            conv = BatchNormalization()(conv)
            conv = tf.keras.activations.swish(conv)
            conv = Conv1D(filters=d_model, kernel_size=1, padding='same')(conv)
            x = x + Dropout(dropout_rate)(conv)
            if not layer_norm_in_front:
                x = LayerNormalization()(x)
        else:
            x = conv

        # Second FF
        if ff_factor1 > 0:
            ff = x
            if layer_norm_in_front:
                ff = LayerNormalization()(ff)

            ff = Conv1D(int(ff_multiplier * d_model), ff_kernel_size,
                        padding='same', activation=activation)(x)
            ff = Dropout(dropout_rate)(ff)
            ff = Conv1D(d_model, ff_kernel_size, padding='same')(ff)
            ff = Dropout(dropout_rate)(ff)
            x = x + ff_factor1 * ff

            if not layer_norm_in_front:
                x = LayerNormalization()(x)

        return x
    return attention_block


"""                 OTHER BLOCKS                 """
def identity_block(model_config: dict):
    def identity(inputs):
        return inputs
    return identity

