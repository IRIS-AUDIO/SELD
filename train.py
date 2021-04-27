import os
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

import layers
import losses
import models
from data_loader import *
from metrics import * 
from params import get_param
from transforms import *
from utils import adaptive_clip_grad
from utils import get_device
import pdb

@tf.function
def trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer, agc):
    with tf.GradientTape() as tape:
        y_p = model(x, training=True)
        sloss = sed_loss(y[0], y_p[0])
        dloss = doa_loss(y[1], y_p[1])
        
        loss = sloss * loss_weight[0] + dloss * loss_weight[1]

    grad = tape.gradient(loss, model.trainable_variables)
    if agc:
        grad = adaptive_clip_grad(model.trainable_variables, grad)
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
    from tqdm import tqdm
    with tqdm(dataset) as pbar:
        for x, y in pbar:
            if mode == 'train':
                preds, sloss, dloss = trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer, config.agc)
            else:
                preds, sloss, dloss = teststep(model, x, y, sed_loss, doa_loss)

            metric_class.update_states(y, preds)
            metric_values = metric_class.result()
            seld_score = calculate_seld_score(metric_values)

            ssloss(sloss)
            ddloss(dloss)
            ER(metric_values[0])
            F(metric_values[1]*100)
            DER(metric_values[2])
            DERF(metric_values[3]*100)
            SeldScore(seld_score)
            pbar.set_postfix(OrderedDict({
                             'mode' : mode,
                             'epoch' : epoch, 
                             'ErrorRate' : ER.result().numpy(), 
                             'F' : F.result().numpy(), 
                             'sedLoss' : ssloss.result().numpy(),
                             'doaLoss' : ddloss.result().numpy(),
                             'seldScore' : SeldScore.result().numpy()
                             }))

    recall, precision = metric_class.class_result()

    writer.add_scalar(f'{mode}/{mode}_ErrorRate', ER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_F', F.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', 
                      DER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', 
                      DERF.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_sedLoss', 
                      ssloss.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_doaLoss', 
                      ddloss.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_seldScore', 
                      SeldScore.result().numpy(), epoch)
                      
    writer.add_scalars(f'{mode}/{mode}_class_recall', {
    'alarm': recall[0].numpy(),
    'crying_baby': recall[1].numpy(),
    'crash': recall[2].numpy(),
    'barking dog': recall[3].numpy(),
    'running engine': recall[4].numpy(),
    'female scream': recall[5].numpy(),
    'femael speech': recall[6].numpy(),
    'burning fire': recall[7].numpy(),
    'footsteps': recall[8].numpy(),
    'knocking on door': recall[9].numpy(),
    'man scream': recall[10].numpy(),
    'man speech': recall[11].numpy(),
    'ringing phone': recall[12].numpy(),
    'piano': recall[13].numpy(),
    }, epoch)

    writer.add_scalars(f'{mode}/{mode}_class_precision', {
    'alarm': precision[0].numpy(),
    'crying_baby': precision[1].numpy(),
    'crash': precision[2].numpy(),
    'barking dog': precision[3].numpy(),
    'running engine': precision[4].numpy(),
    'female scream': precision[5].numpy(),
    'femael speech': precision[6].numpy(),
    'burning fire': precision[7].numpy(),
    'footsteps': precision[8].numpy(),
    'knocking on door': precision[9].numpy(),
    'man scream': precision[10].numpy(),
    'man speech': precision[11].numpy(),
    'ringing phone': precision[12].numpy(),
    'piano': precision[13].numpy(),
    }, epoch)

    return SeldScore.result()


def get_dataset(config, mode:str='train'):
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'), 
                             mode=mode, n_freq_bins=64)
    # mic_x, _ = load_seldnet_data(os.path.join(path, 'mic_dev_norm'),
    #                          os.path.join(path, 'mic_dev_label'), 
    #                          mode=mode, n_freq_bins=64)
    # x = np.concatenate([x, mic_x], -1)
    
    if mode == 'train' and not 'nomask' in config.name:
        sample_transforms = [
            lambda x, y: (mask(x, axis=-3, max_mask_size=config.time_mask_size, n_mask=6), y),
            lambda x, y: (mask(x, axis=-2, max_mask_size=config.freq_mask_size), y),
        ]
    else:
        sample_transforms = []
    batch_transforms = [split_total_labels_to_sed_doa]
    if config.foa_aug and mode == 'train':
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


def get_tdm_dataset(config, mode:str='train'):
    mode = 'foa'
    abspath = '/media/data1/datasets/DCASE2020' if os.path.exists('/media/data1/datasets') else '/root/datasets/DCASE2020'
    FEATURE_PATH = os.path.join(abspath, f'{mode}_dev')
    LABEL_PATH = os.path.join(abspath, 'metadata_dev')
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    TDM_PATH = './'
    x, y = get_preprocessed_wave(FEATURE_PATH,
                                 LABEL_PATH)

    # 데이터 훑어서 aug용 데이터 뽑기
    # 아래 코드들은 어차피 feature extractor부터 사용해야해서 파이토치로 짜도 무방할듯
    tdm_x, tdm_y, sr = get_TDMset(TDM_PATH)
    x, y = TDM_aug(x, y, tdm_x, tdm_y, sr)
    x = [get_preprocessed_x(x_, sr, mode=mode,
                         n_mels=64, 
                         multiplier=5,
                         max_label_length=600,
                         win_length=960,
                         hop_length=480,
                         n_fft=1024) for x_ in x] 
    x_temp = torch.cat(x, axis=0)
    device = get_device()
    m = torch.from_numpy(np.load('mean.npy')).to(device=device)
    s = torch.from_numpy(np.load('std.npy')).to(device=device)
    x = [(x_ - m)/s for x_ in x]

    x = [tf.convert_to_tensor(x_.cpu().numpy()) for x_ in x]

    if not 'nomask' in config.name:
         sample_transforms = [
            lambda x, y: (mask(x, axis=-3, max_mask_size=config.time_mask_size, n_mask=6), y),
            lambda x, y: (mask(x, axis=-2, max_mask_size=config.freq_mask_size), y),
        ]
    else:
        sample_transforms = []
    # 이후에는 기존 데이터셋 만드는 코드
    # seldnet_data_to_dataloader
    batch_transforms = [split_total_labels_to_sed_doa]
    if config.foa_aug :
        batch_transforms.insert(0, foa_intensity_vec_aug)
    dataset = seldnet_data_to_dataloader(
        x, y,
        train=True,
        batch_transforms=batch_transforms,
        label_window_size=60,
        batch_size=config.batch,
        sample_transforms=sample_transforms,
        loop_time=config.loop_time
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
    # trainset = get_dataset(config, 'train')
    trainset = get_tdm_dataset(config, 'train')
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
    if config.sed_loss == 'BCE':
        sed_loss = tf.keras.losses.BinaryCrossentropy(name='sed_loss')
    if config.sed_loss == 'FOCAL':
        sed_loss = losses.Focal_Loss(alpha=config.focal_g, gamma=config.focal_a)

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
        if config.tdm_epoch:
            trainset = get_tdm_dataset(config, 'train')
            valset = get_dataset(config, 'val')
            testset = get_dataset(config, 'test')

    
    best_score = 99999
    early_stop_patience = 0
    lr_decay_patience = 0
    metric_class = SELDMetrics(
        doa_threshold=config.lad_doa_thresh)

    for epoch in range(config.epoch):
        # tdm
        if config.tdm_epoch != 0 and epoch % config.tdm_epoch == 0:
            trainset = get_tdm_dataset(config, 'train')
            valset = get_dataset(config, 'val')
            testset = get_dataset(config, 'test')
            
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
            early_stop_patience = 0
            lr_decay_patience = 0
            tf.keras.models.save_model(
                model, 
                os.path.join(model_path, f'bestscore_{best_score}.hdf5'), 
                include_optimizer=False)
        else:
            if lr_decay_patience == config.lr_patience and config.decay != 1:
                optimizer.learning_rate = optimizer.learning_rate * config.decay
                print(f'lr: {optimizer.learning_rate.numpy()}')
                lr_decay_patience = 0
            if early_stop_patience == config.patience:
                print(f'Early Stopping at {epoch}, score is {score}')
                break
            early_stop_patience += 1
            lr_decay_patience += 1


if __name__=='__main__':
    main(get_param())

