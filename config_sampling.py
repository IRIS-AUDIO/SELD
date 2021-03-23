import copy
import random
from functools import reduce
from collections import OrderedDict


def config_sampling(search_space: OrderedDict):
    sample = copy.deepcopy(search_space)

    # key must be sorted first
    # block type must be sampled first and its arguments later
    for key in sample.keys():
        if not key.endswith('_ARGS'):
            sample[key] = random.sample(sample[key], 1)[0]
        else:
            block_type = key.replace('_ARGS', '')
            sample[key] = config_sampling(sample[key][sample[block_type]])

    return sample


def complexity(model_config: OrderedDict, 
               input_shape,
               mapping_dict: dict):
    block = None
    total_complexity = {} 

    for key in model_config.keys():
        if block is None:
            block = model_config[key]
        else:
            complexity, output_shape = mapping_dict[block](model_config[key], 
                                                           input_shape)
            total_complexity = dict_add(total_complexity, complexity)
            input_shape = output_shape
            block = None

    return total_complexity


def dict_add(first: dict, second: dict):
    output = copy.deepcopy(first)

    for key in second.keys():
        if key in output:
            output[key] += second[key]
        else:
            output[key] = second[key]

    return output


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


# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
# 
# https://github.com/facebookresearch/pycls/blob/master/pycls/models/blocks.py
def conv2d_complexity(input_shape: list, 
                      filters,
                      kernel_size,
                      strides=(1, 1),
                      groups=1,
                      use_bias=True):
    kernel_size = safe_tuple(kernel_size, 2)
    strides = safe_tuple(strides, 2)

    h, w, c = input_shape[-3:]
    h, w = (h-1)//strides[0] + 1, (w-1)//strides[1] + 1
    new_shape = input_shape[:-3] + [h, w, filters]

    kernel = kernel_size[0] * kernel_size[1]
    flops = kernel * c * filters * h * w // groups
    params = kernel * c * filters // groups
    if use_bias:
        flops += filters 
        params += filters

    return new_shape, {'flops': flops, 'params': params}


def norm_complexity(input_shape, center=True, scale=True):
    return input_shape, {'params': input_shape[-1] * (center + scale)}


def pool2d_complexity(input_shape,
                      pool_size,
                      strides=1):
    strides = safe_tuple(strides, 2)

    h, w, c = input_shape[-3:]
    h, w = (h-1)//strides[0] + 1, (w-1)//strides[1] + 1
    new_shape = input_shape[:-3] + [h, w, c]
    return new_shape, {}


def linear_complexity(input_shape, units, use_bias=True):
    c = input_shape[-1]
    new_shape = input_shape[:-1] + [units]

    size = 1
    for s in input_shape[:-1]:
        size *= s

    flops = (s + use_bias) * c * units
    params = (c + use_bias) * units
    return new_shape, {'flops': flops, 'params': params}

