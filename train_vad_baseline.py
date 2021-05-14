import joblib
import tensorflow as tf
import tqdm

import model_complexity
import layers
import models
from config_sampler import vad_architecture_sampler
from data_loader import *
from utils import dict_add, AdaBelief
from vad_dataloader import get_vad_dataset_from_pairs, preprocess_window

import stage_complexity
from config_sampler import vad_architecture_sampler


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)


def prepare_dataset(pairs, window, batch_size, train=False, n_repeat=1):
    dataset = get_vad_dataset_from_pairs(pairs, window)

    dataset = dataset.repeat(n_repeat)
    if train:
        dataset = dataset.shuffle(len(pairs))

    dataset = data_loader(dataset, loop_time=1, batch_size=batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def train_and_eval(model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   valset: tf.data.Dataset,
                   epochs=1):
    model = models.vad_architecture(input_shape, model_config)

    model.compile(
        optimizer=AdaBelief(0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['AUC', 'binary_accuracy', 'Precision', 'Recall'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                         patience=16,
                                         mode='max'),
        tf.keras.callbacks.ModelCheckpoint('bdnn_baseline.h5',
                                           monitor='val_auc',
                                           save_best_only=True,
                                           mode='max',
                                           verbose=True),
    ]

    history = model.fit(trainset, 
                        epochs=epochs,
                        validation_data=valset,
                        callbacks=callbacks)
    model.load_weights('bdnn_baseline.h5')

    performances = {
        # **history.history,
        **(model_complexity.vad_architecture_complexity(model_config, 
                                                        input_shape)[0])
    }
    return model, performances


def seq_to_windows(seq, window):
    win_size = len(window)
    win_width = tf.reduce_max(window)

    windows = []
    for w in window:
        if w == win_width:
            windows.append(seq[win_width:])
        else:
            windows.append(seq[w:-win_width+w])
    return tf.stack(windows, axis=1)


def windows_to_seq(windows, window):
    win_size = len(window)
    win_width = tf.reduce_max(window)

    total_len = windows.shape[0] + win_width
    
    seq = tf.zeros([total_len, *windows.shape[2:]], dtype=windows.dtype)
    total_counts = tf.zeros([total_len], dtype=windows.dtype)

    for i, w in enumerate(window):
        parts = windows[:, i]
        counts = tf.ones_like(parts)
        paddings = [[w, win_width-w]] + [[0, 0]]*(len(parts.shape)-1)
        
        seq += tf.pad(parts, paddings, 'CONSTANT')
        total_counts += tf.pad(counts, paddings, 'CONSTANT')
    
    return seq / (total_counts + 1e-8)


if __name__=='__main__':
    batch_size = 256

    window = [-19, -10, -1, 0, 1, 10, 19]
    window = preprocess_window(window)
    input_shape = [len(window), 80, 1]

    trainset = prepare_dataset(joblib.load('timit_soundidea_train.jl'),
                               window, batch_size, train=True, n_repeat=8)
    valset = prepare_dataset(joblib.load('libri_aurora_test.jl'),
                             window, batch_size, train=False)
    pairs = joblib.load('libri_aurora_test_tiny.jl')

    # start training
    search_space_2d = {
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
    }
    search_space_1d = {
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

    constraint = sample_constraint(500_000, 600_000)
    default_config = {
        'flatten': False,
        'last_unit': 1,
    }
    model_config = vad_architecture_sampler(
        search_space_2d,
        search_space_1d,
        n_blocks=2,
        input_shape=input_shape,
        default_config=default_config,
        constraint=constraint)

    model, outputs = train_and_eval(model_config, input_shape, 
                                    trainset, valset, epochs=1000)

    ys = []
    ys_hat = []
    for x, y in tqdm.tqdm(pairs):
        x_windows = seq_to_windows(x, window)
        y_hat = model.predict(x_windows, batch_size=batch_size, 
                              use_multiprocessing=True)
        y_hat = windows_to_seq(y_hat, window)

        ys.append(y)
        ys_hat.append(y_hat)
        assert len(y) == len(y_hat)

    ys = tf.concat(ys, axis=0)
    ys_hat = tf.concat(ys_hat, axis=0)

    # calculate metric
    test_auc = tf.keras.metrics.AUC()(ys, ys_hat)
    test_precision = tf.keras.metrics.Precision()(ys, ys_hat)
    test_recall = tf.keras.metrics.Recall()(ys, ys_hat)
    test_f1score = 2 * test_precision * test_recall \
                 / (test_precision + test_recall + 1e-8)

    print(f'test auc: {test_auc:.5f}')
    print(f'test f1score: {test_f1score:.5f}')

