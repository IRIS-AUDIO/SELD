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
                          input_shape,
                          default_config=None,
                          constraint=None):
    '''
    search_space_2d: modules with 2D outputs
    search_space_1d: modules with 1D outputs
    input_shape: (without batch dimension)
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

        if constraint is None or constraint(model_config, input_shape):
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
    import complexity

    search_space_2d = {
        'simple_conv_block': 
            {'filters': [[16], [24], [32], [48], [64], [96], [128], [192], [256]], 
             'pool_size': [[[1, 1]], [[1, 2]], [[1, 4]]]},
        'another_conv_block': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'pool_size': [1, (1, 2), (1, 4)]},
        'res_basic_stage': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'groups': [1, 2, 4, 8, 16, 32, 64]},
        'res_bottleneck_stage': 
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'groups': [1, 2, 4, 8, 16, 32, 64],
             'bottleneck_ratio': [0.25, 0.5, 1, 2, 4, 8]},
        'dense_net_block': 
            {'growth_rate': [4, 6, 8, 12, 16, 24, 32, 48],
             'depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'strides': [1, (1, 2), (1, 4)],
             'bottleneck_ratio': [0.25, 0.5, 1, 2, 4, 8],
             'reduction_ratio': [0.5, 1, 2]},
        'sepformer_block': 
            {'pos_encoding': [None, 'basic', 'rff'],
             'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
             'kernel_size': [1, 3]},
        'xception_basic_block':
            {'filters': [16, 24, 32, 48, 64, 96, 128, 192, 256],
             'strides': [(1, 2)],
             'mid_ratio': [1]},
        'identity_block': 
            {},
    }
    search_space_1d = {
        'bidirectional_GRU_block':
            {'units': [[16], [24], [32], [48], [64], [96], [128], [192], [256]]}, 
        'transformer_encoder_block':
            {'n_head': [1, 2, 4, 8],
             'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
             'kernel_size': [1, 3]},
        'simple_dense_block':
            {'units': [[16], [24], [32], [48], [64], [96], [128], [192], [256]], 
             'dense_activation': [None, 'relu']},
    }

    def sample_constraint(min_flops=None, max_flops=None, 
                          min_params=None, max_params=None):
        # this contraint was designed for conv_temporal
        def _contraint(model_config, input_shape):
            def get_complexity(block_type):
                return getattr(complexity, f'{block_type}_complexity')

            shape = input_shape[-3:]
            total_cx = {}

            total_cx, shape = complexity.conv2d_complexity(
                shape, model_config['filters'], model_config['first_kernel_size'],
                padding='same', prev_cx=total_cx)
            total_cx, shape = complexity.norm_complexity(shape, prev_cx=total_cx)
            total_cx, shape = complexity.pool2d_complexity(
                shape, model_config['first_pool_size'], padding='same', 
                prev_cx=total_cx)

            # main body parts
            blocks = [b for b in model_config.keys()
                      if b.startswith('BLOCK') and not b.endswith('_ARGS')]
            blocks.sort()

            for block in blocks:
                # input shape check
                if model_config[block] not in search_space_1d and len(shape) != 3:
                    return False

                try:
                    cx, shape = get_complexity(model_config[block])(
                        model_config[f'{block}_ARGS'], shape)
                    total_cx = dict_add(total_cx, cx)
                except ValueError as e:
                    return False

            # sed + doa
            try:
                cx, sed_shape = get_complexity(model_config['SED'])(
                    model_config['SED_ARGS'], shape)
                cx, sed_shape = complexity.linear_complexity(
                    sed_shape, model_config['n_classes'], prev_cx=cx)
                total_cx = dict_add(total_cx, cx)

                cx, doa_shape = get_complexity(model_config['DOA'])(
                    model_config['DOA_ARGS'], shape)
                cx, doa_shape = complexity.linear_complexity(
                    doa_shape, 3*model_config['n_classes'], prev_cx=cx)
                total_cx = dict_add(total_cx, cx)
            except ValueError as e:
                return False

            # total complexity contraint
            if min_flops and total_cx['flops'] < min_flops:
                return False
            if max_flops and total_cx['flops'] > max_flops:
                return False
            if min_params and total_cx['params'] < min_params:
                return False
            if max_params and total_cx['params'] > max_params:
                return False
            return True
        return _contraint

    default_config = {
        'filters': 16,
        'first_kernel_size': 5,
        'first_pool_size': [5, 1],
        'n_classes': 14}

    input_shape = [300, 64, 4]
    min_flops, max_flops = 750_000_000, 1_333_333_333

    import models # for test
    import tensorflow.keras.backend as K

    for i in range(100):
        model_config = conv_temporal_sampler(
            search_space_2d,
            search_space_1d,
            n_blocks=4,
            input_shape=input_shape,
            default_config=default_config,
            constraint=sample_constraint(min_flops, max_flops))
        print(complexity.conv_temporal_complexity(model_config, input_shape))

        # for test
        model = models.conv_temporal(input_shape, model_config)
        print(model.output_shape, 
              sum([K.count_params(p) for p in model.trainable_weights]))

