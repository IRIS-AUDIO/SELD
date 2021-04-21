import os
import tensorflow as tf
import argparse
import json

import complexity
import models
from data_loader import *
from metrics import *
from metrics import * 
from transforms import *
from config_sampler import conv_temporal_sampler
from complexity import conv_temporal_complexity
from utils import dict_add


args = argparse.ArgumentParser()

args.add_argument('--abspath', type=str, default='./')
args.add_argument('--json_fname', type=str, default='results.json')
args.add_argument('--n_samples', type=int, default=128)
args.add_argument('--n_blocks', type=int, default=4)
args.add_argument('--min_flops', type=int, default=750_000_000)
args.add_argument('--max_flops', type=int, default=1_333_333_333)

args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--epochs', type=int, default=50)
args.add_argument('--lr', type=int, default=1e-3)
args.add_argument('--n_classes', type=int, default=14)


def get_dataset(config, mode: str = 'train'):
    # assume foa datasets
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, 
                             n_freq_bins=64)
    
    sample_transforms = None
    batch_transforms = [split_total_labels_to_sed_doa]

    if mode == 'train':
        sample_transforms = [
            # time masking
            lambda x, y: (mask(x, axis=-3, max_mask_size=24, n_mask=6), y),
            # freq masking
            lambda x, y: (mask(x, axis=-2, max_mask_size=16), y),
        ]
        batch_transforms.insert(0, foa_intensity_vec_aug)

    dataset = seldnet_data_to_dataloader(
        x, y,
        train=mode == 'train',
        label_window_size=60,
        batch_size=config.batch_size,
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms)

    return dataset


'''            SEARCH SPACES           '''
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
         'key_dim': [4, 6, 8, 12, 16, 24, 32, 48],
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
         'key_dim': [4, 6, 8, 12, 16, 24, 32, 48],
         'ff_multiplier': [0.25, 0.5, 1, 2, 4, 8],
         'kernel_size': [1, 3]},
    'simple_dense_block':
        {'units': [[16], [24], [32], [48], [64], [96], [128], [192], [256]], 
         'dense_activation': [None, 'relu']},
    'conformer_encoder_block':
        {'key_dim': [4, 6, 8, 12, 16, 24, 32, 48],
         'n_head': [1, 2, 4, 8],
         'kernel_size': [16, 24, 32, 48, 64, 96, 128, 192, 256],
         'multiplier': [1, 2, 4, 8],
         'pos_encoding': [None, 'basic', 'rff']},
}


def sample_constraint(min_flops=None, max_flops=None, 
                      min_params=None, max_params=None):
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


def evaluate_model(input_shape,
                   train_config,
                   model_config: dict,
                   trainset: tf.data.Dataset,
                   total_testset,
                   metric_class):
    model = models.conv_temporal(input_shape, model_config)
    optimizer = tf.keras.optimizers.Adam(train_config.lr)
    sed_loss = tf.keras.losses.BinaryCrossentropy()
    doa_loss = tf.keras.losses.MSE

    model.compile(optimizer=optimizer,
                  loss=[sed_loss, doa_loss],
                  loss_weights=[1, 1000])

    history = model.fit(trainset, epochs=train_config.epochs)

    # evaluate model
    metric_class.reset_states()
    for x, y in trainset:
        y_hat = model.predict_on_batch(x)
        metric_class.update_states(y, y_hat)
    train_performance = metric_class.result()

    metric_class.reset_states()
    y_hat = model.predict(total_testset[0], batch_size=train_config.batch_size)
    metric_class.update_states(total_testset[1], y_hat)
    test_performance = metric_class.result()

    metric_class.reset_states()
    performances = {
        'train_er': train_performance[0].numpy().tolist(),
        'train_f': train_performance[1].numpy().tolist(),
        'train_de': train_performance[2].numpy().tolist(),
        'train_de_f': train_performance[3].numpy().tolist(),
        'test_er': test_performance[0].numpy().tolist(),
        'test_f': test_performance[1].numpy().tolist(),
        'test_de': test_performance[2].numpy().tolist(),
        'test_de_f': test_performance[3].numpy().tolist(),
        **history.history,
        **(conv_temporal_complexity(model_config, input_shape)[0])
    }
    del model, optimizer, history

    return performances


if __name__=='__main__':
    train_config = args.parse_args()
    input_shape = [300, 64, 7]

    # prepare datasets
    trainset = get_dataset(train_config, 'train')

    testset = get_dataset(train_config, 'test')
    test_xs, test_ys = [], []
    for x, y in testset:
        test_xs.append(x)
        test_ys.append(y)
    del testset
    test_xs = tf.concat(test_xs, 0)
    test_ys = tuple(zip(*test_ys))
    test_ys = tf.concat(test_ys[0], 0), tf.concat(test_ys[1], 0)
    total_testset = (test_xs, test_ys)

    # prepare model config sampler
    default_config = {
        'filters': 16,
        'first_kernel_size': 5,
        'first_pool_size': [5, 1],
        'n_classes': train_config.n_classes}
    
    # LOOP
    constraint = sample_constraint(train_config.min_flops, 
                                   train_config.max_flops)
    results = {'train_config': vars(train_config)}

    metric_class = SELDMetrics()

    for i in range(train_config.n_samples):
        model_config = conv_temporal_sampler(
            search_space_2d,
            search_space_1d,
            n_blocks=train_config.n_blocks,
            input_shape=input_shape,
            default_config=default_config,
            constraint=constraint)
        outputs = evaluate_model(
            input_shape, 
            train_config, model_config, trainset, total_testset, metric_class)

        results[f'{i:03d}'] = {'config': model_config, 'perf': outputs}
        with open(train_config.json_fname, 'w') as f:
            json.dump(results, f, indent=4)

