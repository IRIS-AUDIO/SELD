import copy
import random


def config_sampling(search_space):
    sample = copy.deepcopy(search_space)

    # key must be sorted first
    # block type must be sampled first and its arguments later
    for key in sorted(sample.keys()):
        if not key.endswith('_ARGS'):
            sample[key] = random.sample(sample[key], 1)[0]
        else:
            block_type = key.replace('_ARGS', '')
            sample[key] = config_sampling(sample[key][sample[block_type]])

    return sample

