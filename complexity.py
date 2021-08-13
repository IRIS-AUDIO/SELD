# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
#
# prev_cx: previous complexity
# 
# references
# https://github.com/facebookresearch/pycls/blob/master/pycls/models/blocks.py
import copy
from utils import *


'''            module complexities            '''
def mother_block_complexity(model_config, input_shape):
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

    # squeeze and excitation
    squeeze_ratio = model_config.get('squeeze_ratio', 0)
    
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

    shapes = [input_shape]
    cx = {}
    
    # first layer
    if filters0 > 0:
        cx, shape = conv2d_complexity(shapes[-1], filters0, kernel_size0, 
                                      padding='same', prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        if connect0[0] == 1:
            skip = shapes[-1]
            if skip[-3:] != shape[-3:]:
                cx, skip = conv2d_complexity(skip, filters0, 1, prev_cx=cx)
                cx, skip = norm_complexity(skip, prev_cx=cx)
    else:
        shape = shapes[-1][:]
    shapes.append(shape)

    # second layer
    if filters1 > 0:
        cx, shape = conv2d_complexity(shapes[-1], filters1, kernel_size1,
                                      padding='same', strides=strides,
                                      prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        for i in range(2):
            if connect1[i] == 1:
                skip = shapes[i]
                if skip[-3:] != shape[-3:]:
                    cx, skip = conv2d_complexity(skip, filters1, 1, 
                                                 strides=strides, prev_cx=cx)
                    cx, skip = norm_complexity(skip, prev_cx=cx)
    else:
        shape = shapes[-1][:-1] + [sum([connect1[i]*shapes[i][-1] 
                                       for i in range(2)])]
    shapes.append(shape)

    if filters2 > 0:
        cx, shape = conv2d_complexity(shapes[-1], filters2, kernel_size2,
                                      padding='same', prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        for i in range(3):
            if connect2[i] == 1:
                skip = shapes[i]
                if skip[-3:] != shape[-3:]:
                    cx, skip = conv2d_complexity(
                        skip, filters2, 1, 
                        strides=(1, 1) if i == 2 else strides, 
                        prev_cx=cx)
                    cx, skip = norm_complexity(skip, prev_cx=cx)
    else:
        for i in range(len(connect2)):
            if connect2[i] == 1:
                skip = shapes[i]
                if connect2[-1] == 1 and tuple(strides) != (1, 1) and i < 2:
                    cx, skip = conv2d_complexity(
                        skip, skip[-1], 1, strides=strides, prev_cx=cx)
        shape = shapes[-1][:-1] + [sum([connect2[i]*shapes[i][-1] 
                                       for i in range(3)])]

    # Squeeze and Excitation
    if squeeze_ratio > 0:
        se_filters = int(squeeze_ratio * shape[-1])

        se_shape = [*shape[:-3], 1, 1, shape[-1]]
        cx, se_shape = conv2d_complexity(
            se_shape, se_filters, 1, prev_cx=cx)
        cx, se_shape = conv2d_complexity(
            se_shape, shape[-1], 1, prev_cx=cx)

    return cx, shape


def bidirectional_GRU_block_complexity(model_config, input_shape):
    units_per_layer = model_config['units']

    shape = force_1d_shape(input_shape)

    cx = {}
    for units in units_per_layer:
        cx, shape = gru_complexity(shape, units, bi=True, prev_cx=cx)
    return cx, shape


def RNN_block_complexity(model_config, input_shape):
    units = model_config['units']

    bidirectional = model_config.get('bidirectional', True)
    merge_mode = model_config.get('merge_mode', 'mul') # mul, concat, ave
    rnn_type = model_config.get('rnn_type', 'GRU')

    shape = force_1d_shape(input_shape)

    if rnn_type == 'GRU':
        cx, shape = gru_complexity(shape, units, bi=bidirectional,
                                   merge_mode=merge_mode)
    else:
        cx, shape = lstm_complexity(shape, units, bi=bidirectional,
                                    merge_mode=merge_mode)
    return cx, shape


def transformer_encoder_block_complexity(model_config, input_shape):
    # mandatory parameters
    n_head = model_config['n_head']
    key_dim = model_config['key_dim']
    ff_multiplier = model_config['ff_multiplier'] # default to 4 
    kernel_size = model_config['kernel_size'] # default to 1

    shape = force_1d_shape(input_shape)

    d_model = shape[-1]
    if d_model < n_head or d_model % n_head:
        raise ValueError('invalid n_head')

    ff_dim = int(ff_multiplier * d_model)
    if ff_dim < 1:
        raise ValueError('invalid ff_multiplier')

    cx = {}
    cx, shape = multi_head_attention_complexity(
        shape, n_head, key_dim, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    cx, shape = conv1d_complexity(shape, ff_dim, kernel_size, prev_cx=cx)
    cx, shape = conv1d_complexity(shape, d_model, kernel_size, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    return cx, shape


def simple_dense_block_complexity(model_config, input_shape):
    # mandatory parameters
    units_per_layer = model_config['units']

    kernel_size = model_config.get('kernel_size', 1)

    shape = force_1d_shape(input_shape)

    cx = {}
    for units in units_per_layer:
        if len(shape) == 1:
            cx, shape = linear_complexity(shape, units, prev_cx=cx)
        else:
            cx, shape = conv1d_complexity(shape, units, kernel_size, prev_cx=cx)
    return cx, shape


def identity_block_complexity(model_config, input_shape):
    return {'flops': 0, 'params': 0}, input_shape


def conformer_encoder_block_complexity(model_config, input_shape):
    time, emb = input_shape
    multiplier = model_config.get('multiplier', 4)
    key_dim = model_config.get('key_dim', 36)
    n_head = model_config.get('n_head', 4)
    kernel_size = model_config.get('kernel_size', 32) # 32 
    pos_mode = model_config.get('pos_mode', 'absolute')
    use_bias = model_config.get('use_bias', True)

    if emb < n_head or emb % n_head:
        raise ValueError('invalid n_head')

    if emb % 2:
        raise ValueError('Input Shape should be even')
        
    # normalization and two dense layer 
    cx, shape = norm_complexity(input_shape, prev_cx=None)
    cx, shape = linear_complexity(shape, emb*multiplier, True, cx)
    cx, shape = linear_complexity(shape, emb, True, cx)

    # Multi Head Attention 
    cx, shape = norm_complexity(shape, prev_cx=cx)
    cx, shape = multi_head_attention_complexity(shape, n_head, key_dim,
                                                key_dim, use_bias=use_bias,
                                                use_relative=(pos_mode=='relative'), prev_cx=cx)
    #Convolution & GLU
    cx, shape = norm_complexity(shape, prev_cx=cx)
    cx, shape = conv1d_complexity(shape, 2*emb, 1, prev_cx=cx)
    shape[-1] = shape[-1]//2

    # Depthwise
    cx, shape = conv1d_complexity(shape, emb, kernel_size, groups=emb,
                                  prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)
    cx, shape = conv1d_complexity(shape, emb, 1, prev_cx=cx)

    cx, shape = norm_complexity(shape, prev_cx=cx)
    cx, shape = linear_complexity(shape, emb*multiplier, True, cx)
    cx, shape = linear_complexity(shape, emb, True, cx)

    cx, shape = norm_complexity(shape, prev_cx=cx)
    return cx, shape

    
'''            basic complexities            '''
def conv1d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=1,
                      padding='same',
                      groups=1,
                      use_bias=True,
                      prev_cx=None):
    t, c = input_shape
    not_same = padding != 'same'
    t = (t - 1 - not_same*(kernel_size-1)) // strides + 1
    shape = [t, filters]
    if t < 1:
        raise ValueError('invalid strides, kernel_size')

    flops = kernel_size * c * filters * t // groups
    params = kernel_size * c * filters // groups
    if use_bias:
        params += filters

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})

    return complexity, shape


def conv2d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=(1, 1),
                      padding='same',
                      groups=1,
                      use_bias=True,
                      prev_cx=None):
    if input_shape[-1] < groups or input_shape[-1] % groups:
        raise ValueError('wrong groups')
    if filters < groups or filters % groups:
        raise ValueError('wrong groups')

    kernel_size = safe_tuple(kernel_size, 2)
    strides = safe_tuple(strides, 2)
    not_same = padding != 'same'

    h, w, c = input_shape
    h = (h - 1 - not_same*(kernel_size[0]-1)) // strides[0] + 1
    w = (w - 1 - not_same*(kernel_size[1]-1)) // strides[1] + 1
    output_shape = [h, w, filters]
    if h < 1 or w < 1:
        raise ValueError('invalid strides, kernel_size')

    kernel = kernel_size[0] * kernel_size[1]
    flops = kernel * c * filters * h * w // groups
    params = kernel * c * filters // groups
    if use_bias:
        params += filters

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})

    return complexity, output_shape


def separable_conv2d_complexity(input_shape: list, 
                                filters,
                                kernel_size,
                                strides=(1, 1),
                                padding='same',
                                depth_multiplier=1,
                                use_bias=True,
                                prev_cx=None):
    cx = prev_cx if prev_cx else {}
    chan = input_shape[-1]
    cx, shape = conv2d_complexity(input_shape,
                                  int(chan*depth_multiplier),
                                  kernel_size,
                                  strides,
                                  padding=padding,
                                  groups=chan,
                                  use_bias=False,
                                  prev_cx=cx)                                  
    cx, shape = conv2d_complexity(shape, filters, 1, use_bias=use_bias,
                                  prev_cx=cx)                                  

    return cx, shape


def norm_complexity(input_shape, center=True, scale=True, prev_cx=None):
    complexity = dict_add(
        {'params': input_shape[-1] * (center + scale)},
        prev_cx if prev_cx else {})
    return complexity, input_shape


def pool2d_complexity(input_shape, pool_size, strides=None, 
                      padding='valid', prev_cx=None):
    if strides is None:
        strides = pool_size
    strides = safe_tuple(strides, 2)
    not_same = padding != 'same'

    h, w, c = input_shape
    h = (h - 1 - not_same*(strides[0]-1)) // strides[0] + 1
    w = (w - 1 - not_same*(strides[1]-1)) // strides[1] + 1
    output_shape = input_shape[:-3] + [h, w, c]
    if h < 1 or w < 1:
        raise ValueError('invalid strides, kernel_size')

    complexity = prev_cx if prev_cx else {}
    return complexity, output_shape


def linear_complexity(input_shape, units, use_bias=True, prev_cx=None):
    c = input_shape[-1]
    output_shape = input_shape[:-1] + [units]

    size = 1
    for s in input_shape[:-1]:
        size *= s

    flops = size * (c + use_bias) * units
    params = (c + use_bias) * units
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


def gru_complexity(input_shape, units, use_bias=True,
                   bi=True, merge_mode='mul', prev_cx=None):
    num_steps, input_chan = input_shape[-2:]

    params = 3 * units * (input_chan + units + 2 * use_bias)
    if bi:
        params *= 2

    # https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/rnn_hooks.py
    flops = num_steps * (units + input_chan + 2 * use_bias + 1) * units * 3
    # flops += units # hadamard product
    if bi:
        flops *= 2

    output_shape = input_shape[:-1] + [units]
    if merge_mode == 'concat':
        output_shape[-1] = units * 2

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


def lstm_complexity(input_shape, units, use_bias=True,
                   bi=True, merge_mode='mul', prev_cx=None):
    num_steps, input_chan = input_shape[-2:]

    params = 4 * units * (input_chan + units + use_bias)
    if bi:
        params *= 2

    # https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/rnn_hooks.py
    flops = num_steps * (units + input_chan + 2 * use_bias + 1) * units * 4
    if bi:
        flops *= 2

    output_shape = input_shape[:-1] + [units]
    if merge_mode == 'concat':
        output_shape[-1] = units * 2

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


def multi_head_attention_complexity(input_shape, num_heads, key_dim, 
                                    value_dim=None,
                                    use_relative=False,
                                    use_bias=True, 
                                    prev_cx=None):
    # It only assume self attention
    c = input_shape[-1]
    size = 1
    for s in input_shape[:-1]:
        size *= s
    if value_dim == None:
        value_dim = key_dim
    # making Q, K, V
    params = num_heads*(c + use_bias)*(key_dim*2 + value_dim)
    
    # positional encoding bias and kernel 
    if use_relative:
        params += num_heads*key_dim*2 + num_heads*key_dim*c

    # Value to output
    params += num_heads*c*value_dim + c*use_bias

    # embedding
    flops = size*num_heads*(2*key_dim*(c + use_bias) + value_dim*(c + use_bias))

    # positional encoding to kernel mapping
    if use_relative:
        flops += size*c*num_heads*key_dim

    # scaled dot product attention & context
    flops += (size*size*key_dim + size*size*value_dim)*num_heads
    
    # scaled dot product for position 
    if use_relative:
        flops += (size*size*key_dim)*num_heads
 
    # context to output size
    flops += size*(value_dim * num_heads + use_bias)*c
    
    output_shape = input_shape
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})

    return complexity, output_shape


if __name__ == '__main__':
    import json

    model_config = json.load(open('model_config/seldnet.json', 'rb'))
    input_shape = [300, 64, 7]

    print(conv_temporal_complexity(model_config, input_shape))

    import tensorflow as tf
    import models
    model = models.conv_temporal(input_shape, model_config)
    print(sum([tf.keras.backend.count_params(p) 
               for p in model.trainable_weights]))

