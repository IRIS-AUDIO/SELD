# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
#
# prev_cx: previous complexity
# 
import copy
from utils import dict_add
from complexity import *


def mother_stage_complexity(model_config, input_shape):
    # mandatory parameters
    depth = model_config['depth']
    strides = model_config['strides']

    model_config = copy.deepcopy(model_config)
    shape = input_shape
    total_cx = {}

    for i in range(depth):
        cx, shape = mother_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
        model_config['strides'] = 1
    return total_cx, shape


def bidirectional_GRU_stage_complexity(model_config, input_shape):
    depth = model_config['depth']
    units = model_config['units']
    model_config = copy.deepcopy(model_config)
    model_config['units'] = [units] * depth

    return bidirectional_GRU_block_complexity(model_config, input_shape)


def RNN_stage_complexity(model_config, input_shape):
    depth = model_config['depth']
    units = model_config['units']

    shape = input_shape
    total_cx = {}

    for i in range(depth):
        cx, shape = RNN_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
    return total_cx, shape


def simple_dense_stage_complexity(model_config, input_shape):
    depth = model_config['depth']
    units = model_config['units']
    model_config = copy.deepcopy(model_config)
    model_config['units'] = [units] * depth
    
    return simple_dense_block_complexity(model_config, input_shape)


def transformer_encoder_stage_complexity(model_config, input_shape):
    depth = model_config['depth']

    shape = input_shape
    total_cx = {}

    shape = force_1d_shape(input_shape)
    for i in range(depth):
        cx, shape = transformer_encoder_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
    return total_cx, shape


def conformer_encoder_stage_complexity(model_config, input_shape):
    depth = model_config['depth']

    shape = input_shape
    total_cx = {}

    shape = force_1d_shape(input_shape)
    for i in range(depth):
        cx, shape = conformer_encoder_block_complexity(model_config, shape)
        total_cx = dict_add(total_cx, cx)
    return total_cx, shape

