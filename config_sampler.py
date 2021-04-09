import copy
import random
from collections import OrderedDict

from utils import dict_add


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


def conv_temporal_sampler(search_space_2d: dict, 
                          search_space_1d: dict,
                          n_blocks: int,
                          default_config=None,
                          constraint=None):
    '''
    search_space_2d: modules with 2D outputs
    search_space_1d: modules with 1D outputs
    default_config: the process will sample model config
                    starting from default_config
                    if not given, it will start from an
                    empty dict
    constraint: func(model_config) -> bool

    assume body parts can take 2D or 1D modules
    + sed, doa parts only take 1D modules
    '''
    search_space_sanity_check(search_space_2d)
    search_space_sanity_check(search_space_1d)

    search_space_total = copy.deepcopy(search_space_2d)
    search_space_total.update(search_space_1d)
    
    modules_total = search_space_total.keys()
    modules_1d = search_space_1d.keys()

    if default_config is None:
        default_config = {}

    i = 0
    while True:
        # body parts
        model_config = copy.deepcopy(default_config)

        for i in range(n_blocks):
            module = random.sample(modules_total, 1)[0]
            model_config[f'BLOCK{i}'] = module
            model_config[f'BLOCK{i}_ARGS'] = {
                k: random.sample(v, 1)[0]
                for k, v in search_space_total[module].items()}

        for head in ['SED', 'DOA']:
            module = random.sample(modules_1d, 1)[0]
            model_config[f'{head}'] = module
            model_config[f'{head}_ARGS'] = {
                k: random.sample(v, 1)[0]
                for k, v in search_space_total[module].items()}

        if constraint is None or constraint(model_config):
            return model_config

        i += 1
        if (i % 1000) == 0:
            print(f'{i}th iters. check constraint')


def search_space_sanity_check(search_space: dict):
    for name in search_space:
        # check whether each value is valid
        for v in search_space[name].values():
            if not isinstance(v, (list, tuple)):
                raise ValueError(f'values of {name} must be tuple or list')
            if len(v) == 0:
                raise ValueError(f'len of value in {name} must be > 0')


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


if __name__ == '__main__':
    search_space_2d = {
        'simple_conv_block': 
            {'filters': [[8], [16], [32], [48], [64], [96], [128]], 
             'pool_size': [[1], [2]]},
        'another_conv_block': 
            {'filters': [8, 12, 16, 24, 32, 48, 64, 96, 128],
             'depth': [1, 2, 3, 4, 5, 6],
             'pool_size': [1, 2]},
        'res_basic_stage': 
            {'filters': [8, 12, 16, 24, 32, 48, 64, 96, 128],
             'depth': [1, 2, 3, 4, 5, 6],
             'strides': [1, 2],
             'groups': [1, 2, 4, 8, 16, 32]},
        'res_bottleneck_stage': 
            {'filters': [8, 12, 16, 24, 32, 48, 64, 96, 128],
             'depth': [1, 2, 3, 4, 5, 6],
             'strides': [1, 2],
             'groups': [1, 2, 4, 8, 16, 32],
             'bottleneck_ratio': [1, 2, 4]},
        'dense_net_block': 
            {'growth_rate': [2, 4, 8, 16, 32, 48],
             'depth': [1, 2, 3, 4, 5, 6],
             'strides': [1, 2],
             'bottleneck_ratio': [0.5, 1, 2, 4],
             'reduction_ratio': [0.5, 1, 2]},
        'sepformer_block': 
            {'pos_encoding': [None, 'basic', 'rff'],
             'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.5, 1, 2, 4, 8, 16],
             'kernel_size': [1, 3, 5]},
        'xception_block':
            {'filters': [8, 16, 32, 48, 64, 96],
             'block_num': [1, 2, 3, 4, 5, 6, 7]},
    }
    search_space_1d = {
        'bidirectional_GRU_block':
            {'units': [[16], [32], [48], [64], [128], [256]]},
        'transformer_encoder_block':
            {'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.5, 1, 2, 4, 8, 16],
             'kernel_size': [1, 3, 5]},
        'simple_dense_block':
            {'units': [[8], [16], [32], [64], [128], [256]]},
        'identity_block': 
            {},
    }
    default_config = {
        'filters': 32,
        'first_pool_size': [5, 2],
        'n_classes': 14}

    def sample_constraint(model_config):
        # if previous module outputs 1D, current module cannot be
        # a module with 2D inputs, outputs
        prev_2d = True
        blocks = [b for b in model_config.keys()
                  if b.startswith('BLOCK') and not b.endswith('_ARGS')]
        blocks.sort()

        for block in blocks:
            if model_config[block] in search_space_1d:
                prev_2d = False
            else: # 2D module
                if not prev_2d:
                    return False

            args = model_config[f'{block}_ARGS']
            if 'groups' in args and 'filters' in args:
                if args['groups'] > args['filters']:
                    return False

        if not prev_2d:
            # assert 1D modules 
            if model_config['SED'] not in search_space_1d:
                return False
            if model_config['DOA'] not in search_space_1d:
                return False

        return True

    model_config = conv_temporal_sampler(search_space_2d,
                                         search_space_1d,
                                         n_blocks=4,
                                         default_config=default_config,
                                         constraint=sample_constraint)
    print(model_config)

    import models
    input_shape = [300, 64, 4]
    model = models.conv_temporal(input_shape, model_config)

