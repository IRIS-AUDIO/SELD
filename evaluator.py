import json
import numpy as np
import os
import tensorflow as tf
import time
from glob import glob
from tqdm import tqdm

import models
from data_loader import load_seldnet_data
from metrics import *
from params import get_param
from transforms import *


def ensemble_outputs(model, xs: list,
                     win_size=300, step_size=5, batch_size=256):
    @tf.function
    def predict(model, x, batch_size):
        windows = tf.signal.frame(x, win_size, step_size, axis=0)

        sed, doa = [], []
        for i in range(int(np.ceil(windows.shape[0]/batch_size))):
            s, d = model(windows[i*batch_size:(i+1)*batch_size], training=False)
            sed.append(s)
            doa.append(d)
        sed = tf.concat(sed, axis=0)
        doa = tf.concat(doa, axis=0)

        # windows to seq
        total_counts = tf.signal.overlap_and_add(
            tf.ones((sed.shape[0], win_size//step_size), dtype=sed.dtype),
            1)[..., tf.newaxis]
        sed = tf.signal.overlap_and_add(tf.transpose(sed, (2, 0, 1)), 1)
        sed = tf.transpose(sed, (1, 0)) / total_counts
        doa = tf.signal.overlap_and_add(tf.transpose(doa, (2, 0, 1)), 1)
        doa = tf.transpose(doa, (1, 0)) / total_counts

        return sed, doa

    # assume 0th dim of each sample is time dim
    seds = []
    doas = []

    for x in xs:
        sed, doa = predict(model, x, batch_size)
        seds.append(sed)
        doas.append(doa)

    return list(zip(seds, doas))


def load_conv_temporal_model(input_shape, model_config, weights):
    with open(model_config, 'r') as o:
        model_config = json.load(o)
    model = models.conv_temporal(input_shape, model_config)
    model.load_weights(weights)
    return model


if __name__ == '__main__':
    CLASS_WISE_EVAL = False

    # loading data
    path = '/datasets/datasets/DCASE2021/feat_label/'
    test_xs, test_ys = load_seldnet_data(
        os.path.join(path, 'foa_dev_norm'),
        os.path.join(path, 'foa_dev_label'),
        mode='test', n_freq_bins=64)
    test_ys = list(map(
        lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

    # loading models
    input_shape = [300, 64, 7]
    n_classes = 12
    saved_models = [
        ['model_config/SS3.json',
         'saved_model/conv_temporal_SS3_MMSE_SS3_agc_swa80_2_v_0/'
         'SWA_best_0.34428465366363525.hdf5'],
        ['model_config/SS5.json',
         'saved_model/conv_temporal_SS5_MMSE_SS5_agc_smt01_mask_v_0/'
         'SWA_best_0.34466397762298584.hdf5'],
        ['model_config/SS5.json',
         'saved_model/conv_temporal_SS5_MSE_SS5_smt01_l21e-4_v_0/'
         'SWA_best_0.34446.hdf5'],
    ]

    # predictions
    outs = []
    for model in saved_models:
        model = load_conv_temporal_model(input_shape, *model)
        outs.append(ensemble_outputs(model, test_xs, batch_size=1024))
        del model
    outs = list(zip(*outs))
    assert len(outs) == len(test_xs)

    # aggregating predictions
    outputs = []
    for out in outs:
        sed, doa = list(zip(*out))
        sed = tf.add_n(sed) / len(sed)
        doa = tf.add_n(doa) / len(doa)
        outputs.append((sed, doa))

    # evaluation
    if CLASS_WISE_EVAL:
        evaluator = SELDMetrics(
            doa_threshold=20, n_classes=1)

        for c in range(n_classes):
            evaluator.reset_states()
            for y, pred in zip(test_ys, outputs):
                y = (y[0][..., c:c+1], y[1][..., c::n_classes])
                pred = (pred[0][..., c:c+1], pred[1][..., c::n_classes])
                evaluator.update_states(y, pred)
            metric_values = evaluator.result()
            seld_score = calculate_seld_score(metric_values).numpy()
            er, f, der, derf = list(map(lambda x: x.numpy(), metric_values))

            print(f'class {c}')
            print(f'ER: {er:4f}, F: {f:4f}, DER: {der:4f}, DERF: {derf:4f}, '
                  f'SELD: {seld_score:4f}')
    else:
        evaluator = SELDMetrics(
            doa_threshold=20, n_classes=n_classes)
        evaluator.reset_states()

        for y, pred in zip(test_ys, outputs):
            evaluator.update_states(y, pred)
        metric_values = evaluator.result()
        seld_score = calculate_seld_score(metric_values).numpy()
        er, f, der, derf = list(map(lambda x: x.numpy(), metric_values))

        print('ensemble outputs')
        print(f'ER: {er:4f}, F: {f:4f}, DER: {der:4f}, DERF: {derf:4f}, '
              f'SELD: {seld_score:4f}')

