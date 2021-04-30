import os
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

import layers
import losses
import models
from data_loader import *
from vad_dataloader import get_vad_dataset_from_pairs
from transforms import *


def prepare_dataset(pairs, window, batch_size, train=False, n_repeat=1):
    dataset = get_vad_dataset_from_pairs(pairs, window)

    if train:
        dataset = dataset.repeat(n_repeat)
        dataset = dataset.shuffle(len(pairs))

    return data_loader(dataset, loop_time=1, batch_size=batch_size)


def train_and_eval(train_config,
                   model_config: dict,
                   input_shape,
                   trainset: tf.data.Dataset,
                   testset: tf.data.Dataset):
    model = models.conv_temporal(input_shape, model_config)
    optimizer = tf.keras.optimizers.Adam(train_config.lr)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MSE,
                  metrics=['auc', 'accuracy'])

    history = model.fit(trainset, testset, epochs=train_config.epochs)
    performances = {
        **history.history,
        **(conv_temporal_complexity(model_config, input_shape)[0])
    }
    del model, optimizer, history
    return performances


if __name__=='__main__':
    import joblib

    train_pairs = joblib.load('timit_soundidea_train.jl')
    test_pairs = joblib.load('libri_aurora_test.jl')

    window = [-19, -10, -1, 0, 1, 10, 19]
    trainset = prepare_dataset(train_pairs, window, 256, train=True, n_repeat=6)
    for x, y in trainset.take(2):
        print(x.shape, y.shape)

