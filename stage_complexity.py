# COMPLEXITY
# 1. assume the last dim is channel dim
# 2. batch dim must be excluded from input_shape
#
# prev_cx: previous complexity
# 
import copy
from utils import dict_add
from complexity import *


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

