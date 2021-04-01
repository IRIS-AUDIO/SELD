# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
#
# prev_cx: previous complexity
# 
# references
# https://github.com/facebookresearch/pycls/blob/master/pycls/models/blocks.py
from utils import dict_add

def res_bottleneck_block_complexity(model_config, input_shape):
    # mandatory parameters
    filters = model_config['filters']
    strides = model_config['strides']
    groups = model_config['groups']
    bottleneck_ratio = model_config['bottleneck_ratio']

    stries = safe_tuple(strides, 2)
    btn_size = int(filters * bottleneck_ratio)

    # calculate
    cx = {}
    cx, output_shape = conv2d_complexity(input_shape, btn_size, 1, prev_cx=cx)
    cx, output_shape = norm_complexity(output_shape, prev_cx=cx)

    cx, output_shape = conv2d_complexity(
        output_shape, btn_size, 3, strides, groups, prev_cx=cx)
    cx, output_shape = norm_complexity(output_shape, prev_cx=cx)

    cx, output_shape = conv2d_complexity(output_shape, filters, 1, prev_cx=cx)
    cx, output_shape = norm_complexity(output_shape, prev_cx=cx)

    if strides != (1, 1) or inputs.shape[-1] != filters:
        cx, output_shape = conv2d_complexity(input_shape, filters, 1, strides, 
                                             prev_cx=cx)
        cx, output_shape = norm_complexity(output_shape, prev_cx=cx)

    return cx, output_shape


''' basic complexities '''
def conv2d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=(1, 1),
                      groups=1,
                      use_bias=True,
                      prev_cx=None):
    kernel_size = safe_tuple(kernel_size, 2)
    strides = safe_tuple(strides, 2)

    h, w, c = input_shape[-3:]
    h, w = (h-1)//strides[0] + 1, (w-1)//strides[1] + 1
    output_shape = input_shape[:-3] + [h, w, filters]

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


def pool2d_complexity(input_shape, pool_size, strides=1, prev_cx=None):
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


def GRU_complexity(input_shape, units, layers, use_bias=True,
                   bi=True, prev_cx=None):
    
    input_chan = input_shape[-1]
    hidden_size = units
    layers = layers
    num_steps = input_shape[-2]
    params = 3*units*(input_chan + units + 1)
    if use_bias:
        params += 3*hidden_size
    if bi:
        params *= 2
    #for flops I refer this part
    #https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/rnn_hooks.py
    flops = 0
    if bi:
        flops += _count_gru_cell(input_chan, hidden_size, use_bias) * 2
    else:
        flops += _count_gru_cell(input_chan, hidden_size, use_bias)
    for i in range(layers - 1):
        if bi:
            flops += _count_gru_cell(hidden_size * 2, hidden_size,
                                         use_bias) * 2
        else:
            flops += _count_gru_cell(hidden_size, hidden_size, use_bias)

    flops *= num_steps
    output_shape = input_shape[:-1] + [units]
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape


def _count_gru_cell(input_size, hidden_size, bias=True):

    total_ops = 0
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    total_ops += hidden_size
    total_ops += hidden_size * 3

    return total_ops


def Attention_complexity(input_shape, num_heads, key_dim, value_dim,
                         use_bias=True, prev_cx=None):
    
    c = input_shape[-1]
    size = 1
    for s in input_shape[:-1]:
        size *= s
    
    # making Q, K, V
    params = num_heads*(c + use_bias)*(key_dim*2 + value_dim)
    
    # Value to output
    params += num_heads*c*value_dim + c*use_bias

    # embedding
    flops = size*num_heads*(2*use_bias + 2*key_dim + value_dim)*c
    
    # scaled dot product attention & context
    flops += (size*size*key_dim + size*size*value_dim)*num_heads

    # context to output size
    flops += size*value_dim*c*num_heads
    
    output_shape = input_shape
    complexity = dict_add(
        {'flops': flops, 'params': params},
        prev_cx if prev_cx else {})
    return complexity, output_shape
# utils
def safe_tuple(tuple_or_scalar, length=2):
    if isinstance(tuple_or_scalar, (int, float)):
        tuple_or_scalar = (tuple_or_scalar, ) * length
    elif isinstance(tuple_or_scalar, (list, tuple)):
        count = len(tuple_or_scalar)
        if count == 1:
            tuple_or_scalar = tuple_or_scalar * length
        elif count != length:
            raise ValueError("length of input must be one or required length")
    return tuple_or_scalar

