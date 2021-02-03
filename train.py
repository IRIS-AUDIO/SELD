import tensorflow as tf
from params import get_param
from models import build_seldnet
from metrics import evaluation_metrics, SELD_evaluation_metrics
from transforms import *
from data_loader import *
import os, pdb

@tf.function
def trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer):
    with tf.GradientTape() as tape:
        y_p = model(x, training=True)
        sloss = sed_loss(y[0], y_p[0])
        dloss = doa_loss(y[1], y_p[1])
        loss = sloss * loss_weight[0] + dloss * loss_weight[1]
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainalbe_variables))

    return y_p, sloss, dloss

@tf.function
def teststep(model, x, y, sed_loss, doa_loss):
    y_p = model(x, training=False)
    sloss = sed_loss(y[0], y_p[0])
    dloss = doa_loss(y[1], y_p[1])
    return y_p, sloss, dloss

def iterloop(model, dataset, sed_loss, doa_loss, metric, config, optimizer=None, mode='train'):
    if mode == 'train':
        for x, y in dataset:
            preds, sloss, dloss = trainstep(model, x, y, sed_loss, doa_loss, [int(i) for i in config.loss_weight.split(',')], optimizer)

    else:
        for x, y in dataset:
            preds, sloss, dloss = teststep(model, x, y, sed_loss, doa_loss)

    pdb.set_trace()
    metric()
    


def get_dataset(config, mode:str='train'):
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    x, y = load_seldnet_data(path+'foa_dev_norm', path+'foa_dev_label', mode='val')

    sample_transforms = [
        lambda x, y: (mask(x, axis=-3, max_mask_size=24, n_mask=6), y),
        lambda x, y: (mask(x, axis=-2, max_mask_size=8), y),
    ]
    batch_transforms = [
        split_total_labels_to_sed_doa
    ]
    dataset = seldnet_data_to_dataloader( # 나중에 batch 조절 가능하도록 수정
        x, y,
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms,
    )
    return dataset

def main(config):

    # data load
    trainset = get_dataset(config, 'train')
    valset = get_dataset(config, 'val')
    testset = get_dataset(config, 'test')
    a = [(i,j) for i,j in trainset.take(1)]
    input_shape = a[0][0].shape
    print('-----------data shape------------')
    print()
    print(f'data shape: {a[0][0].shape}')
    print(f'label shape(sed, doa): {a[0][1][0].shape}, {a[0][1][1].shape}')
    print()
    print('---------------------------------')
    
    class_num = a[0][1][0].shape[-1]
    del a

    # data preprocess

    # model load
    model = build_seldnet(input_shape, n_classes=class_num)

    optimizer = tf.keras.optimizers.Adam()
    sed_loss = tf.keras.losses.BinaryCrossentropy(name='sed_loss')
    doa_loss = tf.keras.losses.MeanSquaredError(name='doa_loss')
    
    # metric
    metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=class_num, doa_threshold=config.lad_doa_thresh)

    for epoch in range(config.epoch):
        # train loop
        iterloop(model, trainset, sed_loss, doa_loss, metric, config, optimizer=optimizer, mode='train') 


        # validation loop


        # evaluation loop


        # tensorboard

if __name__=='__main__':
    import sys
    main(get_param(sys.argv[1:]))