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
from config_sampler import conv_temporal_sampler
from data_loader import *
from metrics import *
from transforms import *
from utils import dict_add

args = argparse.ArgumentParser()

args.add_argument('--json_fname', type=str, required=True)
args.add_argument('--train_path', type=str, 
                  default='/datasets/datasets/DCASE2020/foa_tdm_dev')
args.add_argument('--test_path', type=str, 
                  default='/datasets/datasets/DCASE2020/feat_label')
args.add_argument('--n_samples', type=int, default=256)
args.add_argument('--n_blocks', type=int, default=4)
args.add_argument('--min_flops', type=int, default=200_000_000)
args.add_argument('--max_flops', type=int, default=240_000_000)

args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--n_repeat', type=int, default=10)
args.add_argument('--lr', type=int, default=2e-3)
args.add_argument('--n_classes', type=int, default=14)


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

        shape = input_shape
        total_cx = {}

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


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   testset: tf.data.Dataset,
                   evaluator):
    model = models.conv_temporal(input_shape, model_config)
    optimizer = tf.keras.optimizers.Adam(train_config.lr)

    model.compile(optimizer=optimizer,
                  loss={'sed_out': tf.keras.losses.BinaryCrossentropy(),
                        'doa_out': tf.keras.losses.MSE})

    history = model.fit(trainset,
                        validation_data=testset)

    evaluator.reset_states()
    for x, y in testset:
        evaluator.update_states(y, model(x, training=False))
    scores = evaluator.result()
    scores = {
        'test_error_rate': scores[0].numpy().tolist(),
        'test_f1score': scores[1].numpy().tolist(),
        'test_der': scores[2].numpy().tolist(),
        'test_derf': scores[3].numpy().tolist(),
        'test_seld_score': calculate_seld_score(scores).numpy().tolist(),
    }

    performances = {
        **history.history,
        **scores,
        **(model_complexity.conv_temporal_complexity(model_config, 
                                                     input_shape)[0])
    }
    del model, optimizer, history
    return performances


if __name__=='__main__':
    train_config = args.parse_args()
    input_shape = [300, 64, 7]

    # TRAIN DATASET
    train_x = joblib.load(os.path.join(train_config.train_path, '5_5_x.joblib'))
    train_x = np.concatenate(train_x, axis=0)
    train_y = joblib.load(os.path.join(train_config.train_path, '5_5_y.joblib'))
    train_y = np.concatenate([np.concatenate(y, axis=-1) for y in train_y], axis=0)
    sample_transforms = [
        # time
        lambda x, y: (mask(x, axis=-3, max_mask_size=35), y),
        # freq
        lambda x, y: (mask(x, axis=-2, max_mask_size=24), y),
    ]
    batch_transforms = [foa_intensity_vec_aug, split_total_labels_to_sed_doa]
    trainset = data_loader((train_x, train_y), 
                           sample_transforms=sample_transforms,
                           batch_transforms=batch_transforms,
                           loop_time=train_config.n_repeat,
                           batch_size=train_config.batch_size)
    del train_x, train_y

    # TEST DATASET
    testset = load_seldnet_data(
        os.path.join(train_config.test_path, 'foa_dev_norm'),
        os.path.join(train_config.test_path, 'foa_dev_label'),
        mode='test')
    testset = seldnet_data_to_dataloader(
        *testset, train=False, batch_size=train_config.batch_size,
        batch_transforms=batch_transforms[-1:])

    # Evaluator
    evaluator = SELDMetrics(doa_threshold=20)

    default_config = {
        'n_classes': train_config.n_classes
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
        model_config = conv_temporal_sampler(
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
            trainset, testset, evaluator)
        outputs['time'] = time.time() - start

        results[f'{i:03d}'] = {'config': model_config, 'perf': outputs}
        with open(train_config.json_fname, 'w') as f:
            json.dump(results, f, indent=4)

