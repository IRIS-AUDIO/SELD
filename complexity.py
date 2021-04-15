# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
#
# prev_cx: previous complexity
# 
# references
# https://github.com/facebookresearch/pycls/blob/master/pycls/models/blocks.py
import copy
from utils import dict_add


def simple_conv_block_complexity(model_config, input_shape):
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    shape = input_shape
    cx = {}
    for i in range(len(filters)):
        cx, shape = conv2d_complexity(shape, filters[i], kernel_size=3, 
                                      prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = pool2d_complexity(shape, pool_size[i], prev_cx=cx)
    return cx, shape


def another_conv_block_complexity(model_config, input_shape):
    filters = model_config['filters']
    depth = model_config['depth']
    pool_size = safe_tuple(model_config['pool_size'])

    cx = {}

    for i in range(depth):
        shape = input_shape
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = conv2d_complexity(shape, filters, 3, prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = conv2d_complexity(shape, filters, 3, prev_cx=cx)

        if input_shape[-1] != filters:
            cx, _ = conv2d_complexity(input_shape, filters, 1, prev_cx=cx)
        input_shape = shape

    cx, shape = pool2d_complexity(shape, pool_size, prev_cx=cx)
    return cx, shape


def res_basic_stage_complexity(model_config, input_shape):
    # mandatory parameters
    depth = model_config['depth']
    strides = model_config['strides']

    model_config = copy.deepcopy(model_config)
    shape = input_shape
    total_cx = {}

    for i in range(depth):
        cx, shape = res_basic_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
        model_config['strides'] = 1
    return total_cx, shape


def res_basic_block_complexity(model_config, input_shape):
    # mandatory parameters
    filters = model_config['filters']
    strides = safe_tuple(model_config['strides'])

    groups = model_config.get('groups', 1)

    shape = input_shape
    cx = {}

    cx, shape = conv2d_complexity(shape, filters, 3, strides=strides,
                                  groups=groups, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)
    cx, shape = conv2d_complexity(shape, filters, 3, 
                                  groups=groups, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    if input_shape[-1] != filters:
        cx, _ = conv2d_complexity(input_shape, filters, 1, strides=strides, prev_cx=cx)
        cx, _ = norm_complexity(shape, prev_cx=cx)
    return cx, shape


def res_bottleneck_stage_complexity(model_config, input_shape):
    # mandatory parameters
    depth = model_config['depth']
    strides = model_config['strides']

    model_config = copy.deepcopy(model_config)
    shape = input_shape
    total_cx = {}

    for i in range(depth):
        cx, shape = res_bottleneck_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
        model_config['strides'] = 1
    return total_cx, shape


def res_bottleneck_block_complexity(model_config, input_shape):
    # mandatory parameters
    filters = model_config['filters']
    strides = model_config['strides']

    groups = model_config.get('groups', 1)
    bottleneck_ratio = model_config.get('bottleneck_ratio', 1)

    strides = safe_tuple(strides, 2)
    btn_size = int(filters * bottleneck_ratio)

    # calculate
    cx = {}
    cx, shape = conv2d_complexity(input_shape, btn_size, 1, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    cx, shape = conv2d_complexity(
        shape, btn_size, 3, strides, groups, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    cx, shape = conv2d_complexity(shape, filters, 1, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    if strides != (1, 1) or input_shape[-1] != filters:
        cx, shape = conv2d_complexity(input_shape, filters, 1, strides, 
                                             prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)

    return cx, shape


def dense_net_block_complexity(model_config, input_shape):
    # mandatory
    growth_rate = model_config['growth_rate']
    depth = model_config['depth']
    strides = safe_tuple(model_config['strides'])

    bottleneck_ratio = model_config.get('bottleneck_ratio', 4)
    reduction_ratio = model_config.get('reduction_ratio', 0.5)

    bottleneck_size = int(bottleneck_ratio * growth_rate)

    cx = {}
    for i in range(depth):
        shape = input_shape
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = conv2d_complexity(shape, bottleneck_size, 1, use_bias=False,
                                      prev_cx=cx)
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = conv2d_complexity(shape, growth_rate, 3, prev_cx=cx)
        shape[-1] = shape[-1] + input_shape[-1] # concat

        input_shape = shape

    if strides[0] != 1 or strides[1] != 1:
        cx, shape = norm_complexity(shape, prev_cx=cx)
        cx, shape = conv2d_complexity(shape, int(shape[-1] * reduction_ratio),
                                      1, use_bias=False, prev_cx=cx)
        cx, shape = pool2d_complexity(shape, strides, prev_cx=cx)

    return cx, shape


# TODO: Sepformer
# TODO: Xception - separableconv


def bidirectional_GRU_block_complexity(model_config, input_shape):
    units_per_layer = model_config['units']

    shape = input_shape
    if len(shape) == 3:
        shape = [shape[0], shape[1] * shape[2]]

    cx = {}
    for units in units_per_layer:
        cx, shape = gru_complexity(shape, units, bi=True, prev_cx=cx)
    return cx, shape


def transformer_encoder_block_complexity(model_config, input_shape):
    # mandatory parameters
    n_head = model_config['n_head']
    ff_multiplier = model_config['ff_multiplier'] # default to 4 
    kernel_size = model_config['kernel_size'] # default to 1

    shape = input_shape
    if len(shape) == 3:
        shape = [shape[0], shape[1] * shape[2]]

    d_model = shape[-1]

    cx = {}
    cx, shape = multi_head_attention_complexity(
        shape, n_head, d_model//n_head, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    cx, shape = conv1d_complexity(shape, int(ff_multiplier*d_model),
                                  kernel_size, prev_cx=cx)
    cx, shape = conv1d_complexity(shape, d_model, kernel_size, prev_cx=cx)
    cx, shape = norm_complexity(shape, prev_cx=cx)

    return cx, shape


# TODO: simple_dense_block - n_classes issue must be fixed

def identity_block_complexity(model_config, input_shape):
    return {'flops': 0, 'params': 0}, input_shape


'''            basic complexities            '''
def conv1d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=1,
                      use_bias=True,
                      prev_cx=None):
    t, c = input_shape
    t = (t-1) // strides + 1
    shape = [t, filters]

    flops = kernel_size * c * filters * t
    params = kernel_size * c * filters
    if use_bias:
        flops += filters 
        params += filters

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})

    return complexity, shape


def conv2d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=(1, 1),
                      groups=1,
                      use_bias=True,
                      prev_cx=None):
    kernel_size = safe_tuple(kernel_size, 2)
    strides = safe_tuple(strides, 2)

    h, w, c = input_shape
    h, w = (h-1)//strides[0] + 1, (w-1)//strides[1] + 1
    output_shape = [h, w, filters]

    kernel = kernel_size[0] * kernel_size[1]
    flops = kernel * c * filters * h * w // groups
    params = kernel * c * filters // groups
    if use_bias:
        flops += filters 
        params += filters

    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})

    return complexity, output_shape


def norm_complexity(input_shape, center=True, scale=True, prev_cx=None):
    complexity = dict_add(
        {'params': input_shape[-1] * (center + scale)},
        prev_cx if prev_cx else {})
    return complexity, input_shape


def pool2d_complexity(input_shape, pool_size, strides=None, prev_cx=None):
    if strides is None:
        strides = pool_size
    strides = safe_tuple(strides, 2)

    h, w, c = input_shape[-3:]
    h, w = (h-1)//strides[0] + 1, (w-1)//strides[1] + 1
    output_shape = input_shape[:-3] + [h, w, c]

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
                   bi=True, prev_cx=None):
    
    input_chan = input_shape[-1]
    num_steps = input_shape[-2]
    params = 3 * units * (input_chan + units + 2 * use_bias)
    if bi:
        params *= 2
    #for flops I refer this part
    #https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/rnn_hooks.py
    flops = (units + input_chan + 2 * use_bias +1) * units * 3
    #hadamard product
    flops += units * 4
    if bi:
        flops *= 2
    flops *= num_steps
    output_shape = input_shape[:-1] + [units]
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


def multi_head_attention_complexity(input_shape, num_heads, key_dim, 
                                    value_dim=None,
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
    
    # Value to output
    params += num_heads*c*value_dim + c*use_bias

    # embedding
    flops = size*num_heads*(2*key_dim*(c + use_bias) + value_dim*(c + use_bias))
    
    # scaled dot product attention & context
    flops += (size*size*key_dim + size*size*value_dim)*num_heads

    # context to output size
    flops += size*(value_dim * num_heads + use_bias)*c
    
    output_shape = input_shape
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


# utils
def safe_tuple(tuple_or_scalar, length=2):
    if isinstance(tuple_or_scalar, (int, float)):
        tuple_or_scalar = (tuple_or_scalar, ) * length

    tuple_or_scalar = tuple(tuple_or_scalar)
    count = len(tuple_or_scalar)
    if count == 1:
        tuple_or_scalar = tuple_or_scalar * length
    elif count != length:
        raise ValueError("length of input must be one or required length")
    return tuple_or_scalar

