import argparse
import copy
import joblib
import json
import time
import tensorflow as tf
import tensorflow.keras.backend as K

import layers
import model_complexity
import models
import stage_complexity
from config_sampler import vad_architecture_sampler
from data_loader import *
from utils import dict_add
from vad_dataloader import get_vad_dataset_from_pairs

args = argparse.ArgumentParser()

args.add_argument('--json_fname', type=str, default='vad_results.json')
args.add_argument('--n_samples', type=int, default=256)
args.add_argument('--n_blocks', type=int, default=3)
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
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=10240)])
    except RuntimeError as e:
        print(e)


'''            SEARCH SPACES           '''
search_space_2d = {
    'mother_stage':
        {'depth': [1, 2, 3],
         'filters0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'filters1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'filters2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                      3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'kernel_size0': [1, 3, 5],
         'kernel_size1': [1, 3, 5],
         'kernel_size2': [1, 3, 5],
         'connect0': [[0], [1]],
         'connect1': [[0, 0], [0, 1], [1, 0], [1, 1]],
         'connect2': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
         'strides': [(1, 1), (1, 2), (1, 3)]},
}
search_space_1d = {
    'simple_dense_stage':
        {'depth': [1, 2, 3],
         'units': [3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
         'dense_activation': [None, 'relu'],
         'dropout_rate': [0., 0.2, 0.5]},
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
            try:
                cx, shape = get_complexity(model_config[block])(
                    model_config[f'{block}_ARGS'], shape)
                total_cx = dict_add(total_cx, cx)

                if model_config[block] == 'mother_stage':
                    args = model_config[f'{block}_ARGS']
                    n_convs = ((args['filters0'] > 0)
                               + (args['filters1'] > 0)
                               + (args['filters2'] > 0))

                    if n_convs == 1:
                        if args['filters1'] == 0:
                            return False
                    elif n_convs == 2:
                        if args['filters1'] > 0 \
                                and list(args['strides']) == [1, 1]:
                            return False

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


def postprocess_fn(model_config):
    model_config = copy.deepcopy(model_config)
    blocks = sorted([key for key in model_config.keys()
                     if key.startswith('BLOCK') and not key.endswith('_ARGS')])

    for block in blocks:
        stage_type = model_config[block]
        
        if stage_type == 'mother_stage':
            args = model_config[f'{block}_ARGS']
            if args['filters2'] == 0:
                if args['filters1'] != 0:
                    args['connect2'][2] = 1
                elif args['filters0'] != 0:
                    args['connect2'][1] = 1

            if args['filters0'] == 0:
                args['kernel_size0'] = 0
                args['connect1'][1] = 0
                args['connect2'][1] = 0
            if args['filters1'] == 0:
                args['kernel_size1'] = 0
                args['connect2'][2] = 0
                args['strides'] = [1, 1]
            if args['filters2'] == 0:
                args['kernel_size2'] = 0
    return model_config


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
            config_postprocess_fn=postprocess_fn,
            constraint=constraint)

        start = time.time()
        outputs = train_and_eval(
            train_config, model_config, 
            input_shape, 
            trainset, testset)
        outputs['time'] = time.time() - start

        results[f'{i:03d}'] = {'config': model_config, 'perf': outputs}
        with open(train_config.json_fname, 'w') as f:
            json.dump(results, f, indent=4)

