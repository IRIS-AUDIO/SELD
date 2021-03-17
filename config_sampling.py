import copy
import random
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
    complexity = 0 # FLOPs

    for key in model_config.keys():
        if block is None:
            block = model_config[key]
        else:
            flops, output_shape = mapping_dict[block](model_config[key], 
                                                      input_shape)
            complexity += flops
            input_shape = output_shape
            block = None

    return complexity

