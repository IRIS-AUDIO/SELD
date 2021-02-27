import os
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

import layers
import losses
import models
from data_loader import *
from metrics import * 
from params import get_param
from transforms import *


@tf.function
def trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer):
    with tf.GradientTape() as tape:
        y_p = model(x, training=True)
        sloss = sed_loss(y[0], y_p[0])
        dloss = doa_loss(y[1], y_p[1])
        
        loss = sloss * loss_weight[0] + dloss * loss_weight[1]

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return y_p, sloss, dloss
    

@tf.function
def teststep(model, x, y, sed_loss, doa_loss):
    y_p = model(x, training=False)
    sloss = sed_loss(y[0], y_p[0])
    dloss = doa_loss(y[1], y_p[1])
    return y_p, sloss, dloss


def iterloop(model, dataset, sed_loss, doa_loss, metric_class, config, epoch, writer, optimizer=None, mode='train'):
    # metric
    ER = tf.keras.metrics.Mean()
    F = tf.keras.metrics.Mean()
    DER = tf.keras.metrics.Mean()
    DERF = tf.keras.metrics.Mean()
    SeldScore = tf.keras.metrics.Mean()
    ssloss = tf.keras.metrics.Mean()
    ddloss = tf.keras.metrics.Mean()

    loss_weight = [int(i) for i in config.loss_weight.split(',')]
    with tqdm(dataset) as pbar:
        for x, y in pbar:
            if mode == 'train':
                preds, sloss, dloss = trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer)
            else:
                preds, sloss, dloss = teststep(model, x, y, sed_loss, doa_loss)

            metric_class.update_states(y, preds)
            metric_values = metric_class.result()
            seld_score = calculate_seld_score(metric_values)

            ssloss(sloss)
            ddloss(dloss)
            pbar.set_postfix(epoch=epoch, 
                             ErrorRate=metric_values[0].numpy(), 
                             F=metric_values[1].numpy() * 100, 
                             DoaErrorRate=metric_values[2].numpy(), 
                             DoaErrorRateF=metric_values[3].numpy() * 100, 
                             seldScore=seld_score.numpy())
            ER(metric_values[0])
            F(metric_values[1]*100)
            DER(metric_values[2])
            DERF(metric_values[3]*100)
            SeldScore(seld_score)

    print(f'{mode}_sloss: {ssloss.result().numpy()}')
    print(f'{mode}_dloss: {ddloss.result().numpy()}')
    writer.add_scalar(f'{mode}/{mode}_ErrorRate', ER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_F', F.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', 
                      DER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', 
                      DERF.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_seldScore', 
                      SeldScore.result().numpy(), epoch)

    return SeldScore.result()


def get_dataset(config, mode:str='train'):
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)
    if mode == 'train' and not 'nomask' in config.name:
        sample_transforms = [
            lambda x, y: (mask(x, axis=-3, max_mask_size=config.time_mask_size, n_mask=6), y),
            lambda x, y: (mask(x, axis=-2, max_mask_size=config.freq_mask_size), y),
        ]
    else:
        sample_transforms = []
    batch_transforms = [
        split_total_labels_to_sed_doa
    ]
    dataset = seldnet_data_to_dataloader(
        x, y,
        batch_transforms=batch_transforms,
        label_window_size=60,
        batch_size=config.batch,
        sample_transforms=sample_transforms,
        config=config
    )
    return dataset


def main(config):
    config, model_config = config[0], config[1]

    tensorboard_path = os.path.join('./tensorboard_log', config.name)
    if not os.path.exists(tensorboard_path):
        print(f'tensorboard log directory: {tensorboard_path}')
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(logdir=tensorboard_path)

    model_path = os.path.join('./saved_model', config.name)
    if not os.path.exists(model_path):
        print(f'saved model directory: {model_path}')
        os.makedirs(model_path)

    # data load
    trainset = get_dataset(config, 'train')
    valset = get_dataset(config, 'val')
    testset = get_dataset(config, 'test')

    # extract data size
    x, y = [(x, y) for x, y in trainset.take(1)][0]
    input_shape = x.shape
    sed_shape, doa_shape = tf.shape(y[0]), tf.shape(y[1])
    print('-----------data shape------------')
    print()
    print(f'data shape: {input_shape}')
    print(f'label shape(sed, doa): {sed_shape}, {doa_shape}')
    print()
    print('---------------------------------')

    # model load
    model = getattr(models, config.model)(input_shape, model_config)
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    sed_loss = tf.keras.losses.BinaryCrossentropy(name='sed_loss')
    
    try:
        doa_loss = getattr(tf.keras.losses, config.doa_loss)
    except:
        doa_loss = getattr(losses, config.doa_loss)

    if config.resume:
        from glob import glob
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model is not existing, resume fail')
        model = tf.keras.models.load_model(_model_path[0])
    
    best_score = 99999
    patience = 0
    metric_class = SELDMetrics(
        doa_threshold=config.lad_doa_thresh)

    for epoch in range(config.epoch):
        # train loop
        metric_class.reset_states()
        iterloop(model, trainset, sed_loss, doa_loss, metric_class, config, epoch, writer, optimizer=optimizer, mode='train') 

        # validation loop
        metric_class.reset_states()
        score = iterloop(model, valset, sed_loss, doa_loss, metric_class, config, epoch, writer, mode='val')

        # evaluation loop
        metric_class.reset_states()
        iterloop(model, testset, sed_loss, doa_loss, metric_class, config, epoch, writer, mode='test')

        if best_score > score:
            os.system(f'rm -rf {model_path}/bestscore_{best_score}.hdf5')
            best_score = score
            patience = 0
            tf.keras.models.save_model(
                model, 
                os.path.join(model_path, f'bestscore_{best_score}.hdf5'), 
                include_optimizer=False)
        else:
            if patience == config.patience:
                print(f'Early Stopping at {epoch}, score is {score}')
                break
            patience += 1


if __name__=='__main__':
    import sys
    main(get_param(sys.argv[1:]))

