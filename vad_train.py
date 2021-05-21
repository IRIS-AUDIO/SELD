import argparse
import joblib
import json
import tensorflow as tf
import tensorflow.keras.backend as K

import stage_complexity
import model_complexity
import layers
import models
from config_sampler import vad_architecture_sampler
from data_loader import *
from utils import dict_add
from vad_dataloader import get_vad_dataset_from_pairs

args = argparse.ArgumentParser()

args.add_argument('--json_fname', type=str, default='vad_results.json')
args.add_argument('--n_samples', type=int, default=256)
args.add_argument('--n_blocks', type=int, default=2)
args.add_argument('--min_flops', type=int, default=500_000)
args.add_argument('--max_flops', type=int, default=600_000)

args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--n_repeat', type=int, default=50)
args.add_argument('--lr', type=int, default=1e-3)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)


'''            SEARCH SPACES           '''
search_space_2d = {
    'res_basic_stage':
        {'filters': [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64],
         'depth': [1, 2],
         'strides': [(1, 1)],
         'groups': [1]},
    'res_bottleneck_stage':
        {'filters': [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64],
         'depth': [1, 2],
         'strides': [(1, 1), (1, 2)],
         'groups': [1],
         'bottleneck_ratio': [0.25, 0.35, 0.5, 0.7, 1, 1.41, 2, 2.83, 4]},
}
'''
search_space_2d = {
    'simple_conv_stage': 
        {'filters': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'depth': [1, 2, 3],
         'pool_size': [(1, 1), (1, 2), (1, 3)]},
    'another_conv_stage': 
        {'filters': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'depth': [1, 2, 3],
         'pool_size': [(1, 1), (1, 2), (1, 3)]},
    'res_basic_stage': 
        {'filters': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'depth': [1, 2, 3],
         'strides': [(1, 1), (1, 2), (1, 3)],
         'groups': [1, 2, 4, 8, 16, 32, 64]},
    'res_bottleneck_stage': 
        {'filters': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'depth': [1, 2, 3],
         'strides': [(1, 1), (1, 2), (1, 3)],
         'groups': [1, 2, 4, 8, 16, 32, 64],
         'bottleneck_ratio': [0.25, 0.5, 1, 2, 4]},
    'dense_net_stage': 
        {'growth_rate': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
         'depth': [1, 2, 3],
         'strides': [(1, 1), (1, 2), (1, 4)],
         'bottleneck_ratio': [0.25, 0.5, 1, 2, 4],
         'reduction_ratio': [0.5, 1, 2]},
    'sepformer_stage': 
        {'depth': [1, 2, 3],
         'pos_encoding': [None, 'basic', 'rff'],
         'n_head': [1, 2, 4, 8, 16],
         'key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
         'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
         'kernel_size': [1, 3, 5]},
    'xception_basic_stage':
        {'filters': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'depth': [1, 2, 3],
         'strides': [(1, 1), (1, 2), (1, 3)],
         'mid_ratio': [0.5, 1, 2, 4]},
    'identity_block': 
        {},
}
'''
search_space_1d = {
    'bidirectional_GRU_stage':
        {'depth': [1, 2, 3],
         'units': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]}, 
    'transformer_encoder_stage':
        {'depth': [1, 2, 3],
         'n_head': [1, 2, 4, 8, 16],
         'key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
         'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
         'kernel_size': [1, 3, 5]},
    'simple_dense_stage':
        {'depth': [1, 2, 3],
         'units': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'dense_activation': [None, 'relu'],
         'dropout_rate': [0., 0.2, 0.5]},
    'conformer_encoder_stage':
        {'depth': [1, 2, 3],
         'key_dim': [2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
         'n_head': [1, 2, 4, 8, 16],
         'kernel_size': [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'multiplier': [1, 2, 4],
         'pos_encoding': [None, 'basic', 'rff']},
}


def sample_constraint(min_flops=None, max_flops=None, 
                      min_params=None, max_params=None):
    def _contraint(model_config, input_shape):
        def get_complexity(block_type):
            return getattr(stage_complexity, f'{block_type}_complexity')

        if model_config['flatten']:
            shape = [np.prod(input_shape)]
        else:
            shape = input_shape
        total_cx = {}

        # main body parts
        blocks = sorted([b for b in model_config.keys()
                         if b.startswith('BLOCK') and not b.endswith('_ARGS')])

        for block in blocks:
            if model_config[block] not in search_space_1d and len(shape) != 3:
                return False

            try:
                cx, shape = get_complexity(model_config[block])(
                    model_config[f'{block}_ARGS'], shape)
                total_cx = dict_add(total_cx, cx)
            except ValueError as e:
                return False

        if model_config['BLOCK0'] != 'res_basic_stage':
            return False
        if model_config['BLOCK1'] != 'res_bottleneck_stage':
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


def prepare_dataset(pairs, window, batch_size, train=False, n_repeat=1):
    dataset = get_vad_dataset_from_pairs(pairs, window)

    if train:
        dataset = dataset.repeat(n_repeat)
        dataset = dataset.shuffle(len(pairs))

    dataset = data_loader(dataset, loop_time=1, batch_size=batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   testset: tf.data.Dataset):
    model = models.vad_architecture(input_shape, model_config)
    optimizer = tf.keras.optimizers.Adam(train_config.lr)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(), # MSE,
                  metrics=['AUC', 'binary_accuracy', 'Precision', 'Recall'])

    history = model.fit(trainset, 
                        validation_data=testset)

    performances = {
        **history.history,
        **(model_complexity.vad_architecture_complexity(model_config, 
                                                        input_shape)[0])
    }
    del model, optimizer, history
    return performances


if __name__=='__main__':
    train_config = args.parse_args()

    window = [-19, -10, -1, 0, 1, 10, 19]
    input_shape = [len(window), 80, 1]

    trainset = prepare_dataset(joblib.load('timit_soundidea_train.jl'),
                               window, 
                               train_config.batch_size, 
                               train=True, 
                               n_repeat=train_config.n_repeat)
    testset = prepare_dataset(joblib.load('libri_aurora_test.jl'),
                              window, train_config.batch_size, train=False)

    default_config = {
        'flatten': False,
        'last_unit': 1,
    }
    constraint = sample_constraint(train_config.min_flops, 
                                   train_config.max_flops)
    results = {'train_config': vars(train_config)}
    start_idx = 0

    # resume past results
    if os.path.exists(train_config.json_fname):
        with open(train_config.json_fname, 'r') as f:
            prev_results = json.load(f)

        if results['train_config'] != prev_results['train_config']:
            raise ValueError('prev config has different train_config')
        
        results = prev_results
        start_idx = 1 + max([int(k) for k in results.keys() if k.isdigit()])

    # start training
    for i in range(start_idx, train_config.n_samples):
        model_config = vad_architecture_sampler(
            search_space_2d,
            search_space_1d,
            n_blocks=train_config.n_blocks,
            input_shape=input_shape,
            default_config=default_config,
            constraint=constraint)
        '''
        model_config = {
            'flatten': True,
            'last_unit': len(window),
            'BLOCK0': 'simple_dense_stage',
            'BLOCK0_ARGS': {
                'depth': 2,
                'units': 512,
                'activation': 'relu',
                'dropout_rate': 0.2,
            }
        }
        '''
        outputs = train_and_eval(
            train_config, model_config, 
            input_shape, 
            trainset, testset)

        results[f'{i:03d}'] = {'config': model_config, 'perf': outputs}
        with open(train_config.json_fname, 'w') as f:
            json.dump(results, f, indent=4)

