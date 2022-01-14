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
from utils import write_answer
from utils import load_output_format_file
from utils import segment_labels
from utils import convert_output_format_cartesian_to_polar
from utils import convert_output_format_polar_to_cartesian
from SELD_evaluation_metrics import SELDMetrics_
import joblib


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


def load_test_data(feat_path, n_freq_bins):
    features = sorted(glob(os.path.join(feat_path, '*.npy')))
    features = [np.load(f).astype('float32') for f in features]
    if len(features[0].shape) == 2:
        def extract(x):
            x = np.reshape(x, (x.shape[0], -1, n_freq_bins))
            return x.transpose(0, 2, 1)

        features = list(map(extract, features))
    else:
        # already in shape of [time, freq, chan]
        pass
    return features    


if __name__ == '__main__':    
    from multiprocessing import Pool
    temp = tf.constant([0.])
    # loading data
    mode = 'val'
    # DATASET PATH
    path = '/dataset/DCASE2020/DCASE2021/feat_label/'

    test_xs, test_ys = load_seldnet_data(
        os.path.join(path, 'foa_dev_norm'),
        os.path.join(path, 'foa_dev_label'),
        mode=mode, n_freq_bins=64)
    test_ys = list(map(
        lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

    # loading models
    input_shape = [300, 64, 7]
    n_classes = 12
    saved_models = [
        ['/root/backup/weights_and_configs/SS5.json',
         '/root/backup/weights_and_configs/conv_temporal_SS5_MMSE_best_agc0.02_smooth0.2_v_0/'
         'SWA_best_0.30253.hdf5'],
        ['/root/backup/weights_and_configs/SS5.json',
         '/root/backup/weights_and_configs/conv_temporal_SS5_MMSE_best_agc0.02_v_0/'
         'SWA_best_0.31268.hdf5'],
        #['/root/backup/weights_and_configs/SS5.json',
        # '/root/backup/weights_and_configs/conv_temporal_SS5_MSE_und+_no_tm_smt02_l21e-4_decay05_v_0/'
        # 'SWA_best_0.31787.hdf5'],
        #['/root/backup/weights_and_configs/SS5_t.json',
        # '/root/backup/weights_and_configs/conv_temporal_SS5_t_MMSE_und+_no_tm_smt02_l21e-4_decay05_v_0/'
        # 'SWA_best_0.32151.hdf5'],
        #['/root/backup/weights_and_configs/SS5_t.json',
        # '/root/backup/weights_and_configs/conv_temporal_SS5_t_MMSE_und+_no_tm_smt02_l21e-4_decay025_SWA100_80_v_0/'
        # 'SWA_best_0.32157.hdf5'],
    ]

    # making answer
    output_path = './make_answer/' #output path
    ans_path =  '/dataset/DCASE2020/metadata_dev/'
    
    # Search or not
    outs = []
    for model in saved_models:
        model = load_conv_temporal_model(input_shape, *model)
        outs.append(ensemble_outputs(model, test_xs, batch_size=512))
        del model
    # outs = list(zip(*outs))
    # assert len(outs) == len(test_xs)
    from multiprocessing import Pool
    import functools

    def results(outs, mode, path, output_path, ans_path):
        # threshold = outs[1]
        # unique = outs[2]
        # outs = outs[0] 
        outputs = []
        outs = list(zip(*outs))
        for out in outs:
            sed, doa = list(zip(*out))
            sed = tf.add_n(sed) / len(sed)
            doa = tf.add_n(doa) / len(doa)
            outputs.append((sed, doa))
        base_threshold = [0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7]
        #threshold = [base_threshold[item] for item in threshold]

        splits = {
            'train': [1, 2, 3, 4],
            'val': [5],
            'test': [6]
        }
        label_list = sorted(glob(ans_path + 'dev-' + mode + '/*'))
        label_list = [os.path.split(os.path.splitext(f)[0])[1] for f in label_list if int(f[f.rfind(os.path.sep)+5]) in splits[mode]] 
        seld_ = SELDMetrics_()
        for i, preds in tqdm(enumerate(outputs)):
            answer_class = preds[0] > 0.5
            answer_direction = preds[1]
            write_answer(output_path, label_list[i] + '.csv',answer_class, answer_direction)
            pred = load_output_format_file(os.path.join(output_path,  label_list[i] + '.csv'))
            pred = segment_labels(pred, answer_class.shape[0])

            if mode == 'val':
                gt = load_output_format_file(os.path.join(ans_path + 'dev-val', label_list[i] + '.csv'))
            if mode == 'test':
                gt = load_output_format_file(os.path.join(ans_path + 'dev-test', label_list[i] + '.csv'))

            gt = convert_output_format_polar_to_cartesian(gt)
            gt = segment_labels(gt, answer_class.shape[0])
            seld_.update_seld_scores(pred, gt)
            metric_values = seld_.compute_seld_scores()
            seld_score = calculate_seld_score(metric_values)
            er, f, der, derf = list(map(lambda x: x, metric_values))
            
        return [er, f, der, derf, seld_score]

    result_ = results(mode=mode, path=path, output_path=output_path, ans_path=ans_path, outs=[outs[0]])
    print(result_)
    #with Pool() as p:
    #    result_ = p.map(functools.partial(results, mode=mode, path=path, output_path=output_path, ans_path=ans_path,), outs)

    #result_ = sorted(result_, key = lambda x : x[0])
    #_fid = open('thresh_result.csv', 'w')
    #for item in result_:
    #    _fid.write('{},{},{},{},{},{}\n'.format(str(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])))
