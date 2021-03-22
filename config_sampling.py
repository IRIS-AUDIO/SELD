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

