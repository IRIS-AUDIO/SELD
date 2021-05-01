import copy
import numpy as np

from utils import *
from stage_complexity import *


def conv_temporal_complexity(model_config, input_shape):
    filters = model_config.get('filters', 32)
    first_kernel_size = model_config.get('first_kernel_size', 7)
    first_pool_size = model_config.get('first_pool_size', [5, 1])
    n_classes = model_config.get('n_classes', 14)

    shape = input_shape[-3:]
    total_cx = {}

    total_cx, shape = conv2d_complexity(shape, filters, first_kernel_size,
                                        padding='same', prev_cx=total_cx)
    total_cx, shape = norm_complexity(shape, prev_cx=total_cx)
    total_cx, shape = pool2d_complexity(shape, first_pool_size, padding='same',
                                        prev_cx=total_cx)

    blocks = [key for key in model_config.keys()
              if key.startswith('BLOCK') and not key.endswith('_ARGS')]
    blocks.sort()

    for block in blocks:
        cx, shape = globals()[f'{model_config[block]}_complexity'](
            model_config[f'{block}_ARGS'], shape)
        total_cx = dict_add(total_cx, cx)

    cx, sed_shape = globals()[f'{model_config["SED"]}_complexity'](
        model_config['SED_ARGS'], shape)
    cx, sed_shape = linear_complexity(sed_shape, n_classes, prev_cx=cx)
    total_cx = dict_add(total_cx, cx)

    cx, doa_shape = globals()[f'{model_config["DOA"]}_complexity'](
        model_config['DOA_ARGS'], shape)
    cx, doa_shape = linear_complexity(doa_shape, 3*n_classes, prev_cx=cx)
    total_cx = dict_add(total_cx, cx)

    return total_cx, (sed_shape, doa_shape)


def vad_architecture_complexity(model_config, input_shape):
    flatten = model_config.get('flatten', True)
    last_unit = model_config.get('last_unit', 1)

    shape = [np.prod(input_shape)] if flatten else input_shape
    total_cx = {}

    blocks = sorted([key for key in model_config.keys()
                     if key.startswith('BLOCK') and not key.endswith('_ARGS')])

    for block in blocks:
        cx, shape = globals()[f'{model_config[block]}_complexity'](
            model_config[f'{block}_ARGS'], shape)
        total_cx = dict_add(total_cx, cx)
    total_cx, shape = linear_complexity(shape, last_unit, prev_cx=total_cx)

    return total_cx, shape

