import tensorflow as tf
import numpy as np
import pdb, argparse, json
import layers
import losses
import models
from data_loader import *
from metrics import * 
from params import get_param
from transforms import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_dataset(mode:str='val'):
    path = os.path.join('/root/datasets', 'DCASE2020/feat_label/')
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)

    sample_transforms = []
    batch_transforms = [
        split_total_labels_to_sed_doa
    ]
    dataset = seldnet_data_to_dataloader(
        x, y,
        train= mode == 'train',
        batch_transforms=batch_transforms,
        label_window_size=60,
        batch_size=64,
    )
    return dataset
def MMSE(y_true, y_pred):
    # y_true(doa), y_pred: (batch, time, 42)
    ''' Masked MSE '''
    y_true = tf.cast(y_true, y_pred.dtype)
    sed = tf.reshape(y_true, (*y_true.shape[:-1], 3, -1))
    sed = tf.reduce_sum(sed ** 2, axis=-2)

    sed = tf.round(tf.concat([sed] * 3, axis=-1)) >= 0.5
    sed = tf.cast(sed, y_pred.dtype)
    # print(sed)
    return tf.keras.backend.sqrt(
        tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred) * sed)) \
                / tf.keras.backend.sum(sed)

def masked_mse(y_gt, model_out):
    # y_gt (concat[sedgt, doagt]), model_out(같음): (batch, time, 56)
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    #pdb.set_trace()
    
    # sed_out = tf.keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = tf.concat([sed_out] * 3, axis=-1)
    sed_out = tf.keras.backend.cast(sed_out, 'float32')
    # print(sed_out)
    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_gt[:, :, 14:] - model_out) * sed_out))/tf.keras.backend.sum(sed_out)

tf.random.set_seed(1)
doa_gt = tf.random.uniform(shape = [64, 60, 42])
sed_gt = tf.random.uniform(shape = [64, 60, 14])
Y_pred = tf.random.uniform(shape = [64, 60, 42])
dataset = get_dataset()
x, y = [(x, y) for x, y in dataset.take(1)][0]
input_shape = x.shape
model_config = os.path.join('./model_config', 'seldnet.json')
model_config = argparse.Namespace(**json.load(open(model_config,'rb')))
a_model = models.seldnet(input_shape, model_config)
b_model = tf.keras.models.clone_model(a_model)
optimizer1 = tf.keras.optimizers.Adam()
optimizer2 = tf.keras.optimizers.Adam()
c = 0
for x,y in dataset:
    sed_gt, doa_gt = y
    with tf.GradientTape() as a_tape:
        _, a_Y_pred = a_model(x)
        aa = MMSE(doa_gt, a_Y_pred)
    grad = a_tape.gradient(aa, a_model.trainable_variables)
    optimizer1.apply_gradients(zip(grad, a_model.trainable_variables))
    optimizer2.apply_gradients(zip(grad, b_model.trainable_variables))
    c+= 1
    if c == 6:
        break
a_model.save_weights('test.hdf5')
b_model.load_weights('test.hdf5')
os.system('rm -rf test.hdf5')
count = 0
k = 0
for x,y in dataset:
    sed_gt, doa_gt = y
    pdb.set_trace()

    with tf.GradientTape() as a_tape:
        _, a_Y_pred = a_model(x)
        aa = MMSE(doa_gt, a_Y_pred)
    grad1 = a_tape.gradient(aa, a_model.trainable_variables)
    with tf.GradientTape() as b_tape:
        _, Y_pred = b_model(x)
        bb = masked_mse(tf.concat([sed_gt, doa_gt], -1), Y_pred)
    grad2 = b_tape.gradient(bb, b_model.trainable_variables)

    count += (aa == bb).numpy()
    k += 1
print(count == k)