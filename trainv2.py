import numpy as np
import os
import tensorflow as tf
import time
from collections import OrderedDict
from glob import glob
from numpy import inf
from tensorboardX import SummaryWriter
from tqdm import tqdm

import layers
import losses
import models
from data_loader import *
from metrics import * 
from params import get_param
from swa import SWA
from transforms import *
from utils import adaptive_clip_grad, AdaBelief, apply_kernel_regularizer, write_answer
from utils import write_answer, load_output_format_file, convert_output_format_polar_to_cartesian, segment_labels
from SELD_evaluation_metrics import SELDMetrics_



def generate_trainstep(sed_loss, doa_loss, loss_weights, label_smoothing=0.):
    # These are statistics from the train dataset
    train_samples = tf.convert_to_tensor(
        [[58193, 32794, 29801, 21478, 14822, 
        9174, 66527,  6740,  9342,  6498, 
        22218, 49758]],
        dtype=tf.float32)
    cls_weights = tf.reduce_mean(train_samples) / train_samples
    @tf.function
    def trainstep(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            y_p = model(x, training=True)
            sed, doa = y
            sed_pred, doa_pred = y_p

            if label_smoothing > 0:
                sed = sed * (1-label_smoothing) + 0.5 * label_smoothing

            sloss = tf.reduce_mean(sed_loss(sed, sed_pred) * cls_weights)
            dloss = doa_loss(doa, doa_pred, cls_weights)
            
            loss = sloss * loss_weights[0] + dloss * loss_weights[1]

            # regularizer
            loss += tf.add_n([l.losses[0] for l in model.layers
                              if len(l.losses) > 0])

        grad = tape.gradient(loss, model.trainable_variables)
        # apply AGC
        grad = adaptive_clip_grad(model.trainable_variables, grad)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        return y_p, sloss, dloss
    return trainstep


def generate_teststep(sed_loss, doa_loss):
    @tf.function
    def teststep(model, x, y, optimizer=None):
        y_p = model(x, training=False)
        sloss = sed_loss(y[0], y_p[0])
        dloss = doa_loss(y[1], y_p[1])
        return y_p, sloss, dloss
    return teststep


def generate_iterloop(sed_loss, doa_loss, evaluator, writer, 
                      mode, loss_weights=None):
    if mode == 'train':
        step = generate_trainstep(sed_loss, doa_loss, loss_weights)
    else:
        step = generate_teststep(sed_loss, doa_loss)

    def iterloop(model, dataset, epoch, optimizer=None):
        evaluator.reset_states()
        ssloss = tf.keras.metrics.Mean()
        ddloss = tf.keras.metrics.Mean()

        with tqdm(dataset) as pbar:
            for x, y in pbar:
                preds, sloss, dloss = step(model, x, y, optimizer)

                evaluator.update_states(y, preds)
                metric_values = evaluator.result()
                seld_score = calculate_seld_score(metric_values)

                ssloss(sloss)
                ddloss(dloss)
                pbar.set_postfix(
                    OrderedDict({
                        'mode': mode,
                        'epoch': epoch, 
                        'ER': metric_values[0].numpy(),
                        'F': metric_values[1].numpy(),
                        'DER': metric_values[2].numpy(),
                        'DERF': metric_values[3].numpy(),
                        'seldscore': seld_score.numpy()
                    }))

        writer.add_scalar(f'{mode}/{mode}_ErrorRate', metric_values[0].numpy(),
                          epoch)
        writer.add_scalar(f'{mode}/{mode}_F', metric_values[1].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', 
                          metric_values[2].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', 
                          metric_values[3].numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_sedLoss', 
                          ssloss.result().numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_doaLoss', 
                          ddloss.result().numpy(), epoch)
        writer.add_scalar(f'{mode}/{mode}_seldScore', 
                          seld_score.numpy(), epoch)

        return seld_score.numpy()
    return iterloop


def random_ups_and_downs(x, y):
    x = tf.concat(
        [x[..., :4] + tf.random.normal([], stddev=0.2), x[..., 4:]],
        axis=-1)
    return x, y


def get_dataset(config, mode: str = 'train'):
    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')

    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)
    if config.use_tfm and mode == 'train':
        sample_transforms = [
            random_ups_and_downs,
            lambda x, y: (mask(x, axis=-3, max_mask_size=6, n_mask=10), y),
            lambda x, y: (mask(x, axis=-2, max_mask_size=8, n_mask=6), y),
            # lambda x, y: (mask(x, axis=-3, max_mask_size=12, n_mask=6), y),
            # lambda x, y: (mask(x, axis=-2, max_mask_size=8, n_mask=6), y),
        ]
    else:
        sample_transforms = []
    batch_transforms = [split_total_labels_to_sed_doa]
    if config.use_acs and mode == 'train':
        batch_transforms.insert(0, foa_intensity_vec_aug)
    dataset = seldnet_data_to_dataloader(
        x, y,
        train= mode == 'train',
        batch_transforms=batch_transforms,
        label_window_size=60,
        batch_size=config.batch,
        sample_transforms=sample_transforms,
        loop_time=config.loop_time
    )
    return dataset


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


def generate_evaluate_fn(test_xs, test_ys, evaluator, write_path, ans_path, batch_size=256,
                         writer=None):
    def evaluate_fn(model, epoch):
        start = time.time()
        evaluator.reset_states()
        e_outs = ensemble_outputs(model, test_xs, batch_size=batch_size)

        label_list = sorted(glob(ans_path + '/dev-test' '/*'))
        label_list = [os.path.split(os.path.splitext(f)[0])[1] for f in label_list if int(f[f.rfind(os.path.sep)+5]) in [6]] 
        seld_ = SELDMetrics_()
        for i, preds in tqdm(enumerate(e_outs)):
            answer_class = preds[0] > 0.5
            answer_direction = preds[1]
            write_answer(write_path, label_list[i] + '.csv',answer_class, answer_direction)
            pred = load_output_format_file(os.path.join(write_path,  label_list[i] + '.csv'))
            pred = segment_labels(pred, answer_class.shape[0])
            gt = load_output_format_file(os.path.join(ans_path , 'dev-test', label_list[i] + '.csv'))
            gt = convert_output_format_polar_to_cartesian(gt)
            gt = segment_labels(gt, answer_class.shape[0])
            seld_.update_seld_scores(pred, gt)
            metric_values = seld_.compute_seld_scores()
            seld_score = calculate_seld_score(metric_values)
            er, f, der, derf = list(map(lambda x: x, metric_values))

        #for y, pred in zip(test_ys, e_outs):
        #    evaluator.update_states(y, pred)

        # metric_values = evaluator.result()
        # seld_score = calculate_seld_score(metric_values).numpy()
        er, f, der, derf = list(map(lambda x: x, metric_values))

        if writer is not None:
            writer.add_scalar('ENS_T/ER', er, epoch)
            writer.add_scalar('ENS_T/F', f, epoch)
            writer.add_scalar('ENS_T/DER', der, epoch)
            writer.add_scalar('ENS_T/DERF', derf, epoch)
            writer.add_scalar('ENS_T/seldScore', seld_score, epoch)
        print('ensemble outputs')
        print(f'ER: {er:4f}, F: {f:4f}, DER: {der:4f}, DERF: {derf:4f}, '
              f'SELD: {seld_score:4f} '
              f'({time.time()-start:.4f} secs)')
        return seld_score, metric_values
    return evaluate_fn


def main(config):
    config, model_config = config[0], config[1]

    # HyperParameters
    n_classes = 12
    swa_start_epoch = 80
    swa_freq = 2
    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)

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

    path = os.path.join(config.abspath, 'DCASE2021/feat_label/')
    test_xs, test_ys = load_seldnet_data(
        os.path.join(path, 'foa_dev_norm'),
        os.path.join(path, 'foa_dev_label'), 
        mode='test', n_freq_bins=64)
    test_ys = list(map(
        lambda x: split_total_labels_to_sed_doa(None, x)[-1], test_ys))

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
    model_config['n_classes'] = n_classes
    model = getattr(models, config.model)(input_shape, model_config)
    model.summary()

    model = apply_kernel_regularizer(model, kernel_regularizer)

    optimizer = AdaBelief(config.lr)
    if config.sed_loss == 'BCE':
        sed_loss = tf.keras.backend.binary_crossentropy
    else:
        sed_loss = losses.focal_loss
    # fix doa_loss to MMSE_with_cls_weights (because of class weights)
    doa_loss = losses.MMSE_with_cls_weights

    # stochastic weight averaging
    swa = SWA(model, swa_start_epoch, swa_freq)

    if config.resume:
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model does not exist, cannot be resumed')
        model = tf.keras.models.load_model(_model_path[0])

    best_score = inf
    early_stop_patience = 0
    lr_decay_patience = 0
    evaluator = SELDMetrics(
        doa_threshold=config.lad_doa_thresh, n_classes=n_classes)

    train_iterloop = generate_iterloop(
        sed_loss, doa_loss, evaluator, writer, 'train', 
        loss_weights=list(map(int, config.loss_weight.split(','))))
    val_iterloop = generate_iterloop(
        sed_loss, doa_loss, evaluator, writer, 'val')
    test_iterloop = generate_iterloop(
        sed_loss, doa_loss, evaluator, writer, 'test')
    evaluate_fn = generate_evaluate_fn(
        test_xs, test_ys, evaluator, config.output_path, config.ans_path,  config.batch*4, writer=writer)

    for epoch in range(config.epoch):
        if epoch == swa_start_epoch:
            tf.keras.backend.set_value(optimizer.lr, config.lr * 0.5)

        if epoch % 10 == 0:
            evaluate_fn(model, epoch)

        # train loop
        train_iterloop(model, trainset, epoch, optimizer)
        score = val_iterloop(model, valset, epoch)
        test_iterloop(model, testset, epoch)

        swa.on_epoch_end(epoch)

        if best_score > score:
            os.system(f'rm -rf {model_path}/bestscore_{best_score}.hdf5')
            best_score = score
            early_stop_patience = 0
            lr_decay_patience = 0
            tf.keras.models.save_model(
                model, 
                os.path.join(model_path, f'bestscore_{best_score}.hdf5'), 
                include_optimizer=False)
        else:
            '''
            if lr_decay_patience == config.lr_patience and config.decay != 1:
                optimizer.learning_rate = optimizer.learning_rate * config.decay
                print(f'lr: {optimizer.learning_rate.numpy()}')
                lr_decay_patience = 0
            '''
            if early_stop_patience == config.patience:
                print(f'Early Stopping at {epoch}, score is {score}')
                break
            early_stop_patience += 1
            lr_decay_patience += 1

    # end of training
    print(f'epoch: {epoch}')
    swa.on_train_end()

    seld_score, *_ = evaluate_fn(model, epoch)

    tf.keras.models.save_model(
        model, 
        os.path.join(model_path, f'SWA_best_{seld_score:.5f}.hdf5'),
        include_optimizer=False)


if __name__=='__main__':
    import os
    main(get_param())

