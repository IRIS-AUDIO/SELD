import tensorflow as tf
from params import get_param
from models import build_seldnet
from metrics import evaluation_metrics, SELD_evaluation_metrics
from transforms import *
from data_loader import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, pdb

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

def metric(metric_class, preds, gts, class_num):
    if type(preds[0]) != np.ndarray:
        preds = [i.numpy() for i in preds]
        gts = [i.numpy() for i in gts]
    test_sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(preds[0]) > 0.5
    test_doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(preds[1])
    test_sed_gt = SELD_evaluation_metrics.reshape_3Dto2D(gts[0])
    test_doa_gt = SELD_evaluation_metrics.reshape_3Dto2D(gts[1])

    test_pred_dict = SELD_evaluation_metrics.regression_label_format_to_output_format((test_sed_pred, test_doa_pred), class_num)
    test_gt_dict = SELD_evaluation_metrics.regression_label_format_to_output_format((test_sed_gt, test_doa_gt), class_num)

    test_pred_blocks_dict = SELD_evaluation_metrics.segment_labels(test_pred_dict, test_sed_pred.shape[0])
    test_gt_blocks_dict = SELD_evaluation_metrics.segment_labels(test_gt_dict, test_sed_gt.shape[0])

    metric_class.update_seld_scores_xyz(test_pred_blocks_dict, test_gt_blocks_dict)
    test_new_metric = metric_class.compute_seld_scores()
    test_new_seld_metric = evaluation_metrics.early_stopping_metric(test_new_metric[:2], test_new_metric[2:])
    return test_new_metric, test_new_seld_metric

def iterloop(model, dataset, sed_loss, doa_loss, metric_class, config, class_num, epoch, writer, optimizer=None, mode='train'):
    # metric
    ER = tf.keras.metrics.Mean()
    F = tf.keras.metrics.Mean()
    DER = tf.keras.metrics.Mean()
    DERF = tf.keras.metrics.Mean()
    SeldScore = tf.keras.metrics.Mean()
    ssloss = tf.keras.metrics.Mean()
    ddloss = tf.keras.metrics.Mean()

    with tqdm(dataset) as pbar:
        for x, y in pbar:
            loss_weight = [int(i) for i in config.loss_weight.split(',')]
            if mode == 'train':
                preds, sloss, dloss = trainstep(model, x, y, sed_loss, doa_loss, loss_weight, optimizer)
            else:
                preds, sloss, dloss = teststep(model, x, y, sed_loss, doa_loss)
            test_metric, test_seld_metric = metric(metric_class, preds, y, class_num)
            ssloss(sloss)
            ddloss(dloss)
            pbar.set_postfix(epoch=epoch, 
                                ErrorRate=test_metric[0], 
                                F=test_metric[1]*100, 
                                DoaErrorRate=test_metric[2], 
                                DoaErrorRateF=test_metric[3]*100, 
                                seldScore=test_seld_metric)
            ER(test_metric[0])
            F(test_metric[1]*100)
            DER(test_metric[2])
            DERF(test_metric[3]*100)
            SeldScore(test_seld_metric)
    print(f'{mode}_sloss: {ssloss.result().numpy()}')
    print(f'{mode}_dloss: {ddloss.result().numpy()}')
    writer.add_scalar(f'{mode}/{mode}_ErrorRate', ER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_F', F.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRate', DER.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_DoaErrorRateF', DERF.result().numpy(), epoch)
    writer.add_scalar(f'{mode}/{mode}_seldScore', SeldScore.result().numpy(), epoch)

    return SeldScore.result()


def get_dataset(config, mode:str='train'):
    path = os.path.join(config.abspath, 'DCASE2020/feat_label/')
    time_length = 64
    x, y = load_seldnet_data(path+'foa_dev_norm', path+'foa_dev_label', mode=mode, n_freq_bins=time_length)

    sample_transforms = [
        # lambda x, y: (mask(x, axis=-3, max_mask_size=config.time_mask_size, n_mask=6), y),
        # lambda x, y: (mask(x, axis=-2, max_mask_size=config.freq_mask_size), y),
    ]
    batch_transforms = [
        split_total_labels_to_sed_doa
    ]
    dataset = seldnet_data_to_dataloader( # 나중에 batch 조절 가능하도록 수정
        x, y,
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms,
        label_window_size=time_length,
        batch_size=config.batch
    )
    return dataset

def main(config):
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

    # model load
    model = build_seldnet(input_shape, n_classes=class_num)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    sed_loss = tf.keras.losses.BinaryCrossentropy(name='sed_loss')
    doa_loss = getattr(tf.keras.losses, config.loss)(name='doa_loss')

    if config.resume:
        from glob import glob
        _model_path = sorted(glob(model_path + '/*.hdf5'))
        if len(_model_path) == 0:
            raise ValueError('the model is not existing, resume fail')
        model = tf.keras.models.load_model(_model_path[0])

    
    best_score = 99999
    patience = 0
    for epoch in range(config.epoch):
        # train loop
        metric_class = SELD_evaluation_metrics.SELDMetrics(nb_classes=class_num, doa_threshold=config.lad_doa_thresh)
        iterloop(model, trainset, sed_loss, doa_loss, metric_class, config, class_num, epoch, writer, optimizer=optimizer, mode='train') 

        # validation loop
        metric_class = SELD_evaluation_metrics.SELDMetrics(nb_classes=class_num, doa_threshold=config.lad_doa_thresh)
        score = iterloop(model, trainset, sed_loss, doa_loss, metric_class, config, class_num, epoch, writer, mode='val')

        # evaluation loop
        metric_class = SELD_evaluation_metrics.SELDMetrics(nb_classes=class_num, doa_threshold=config.lad_doa_thresh)
        iterloop(model, trainset, sed_loss, doa_loss, metric_class, config, class_num, epoch, writer, mode='test')

        if best_score > score:
            os.system(f'rm -rf {model_path}/bestscore_{best_score}.hdf5')
            best_score = score
            patience = 0
            tf.keras.models.save_model(model, os.path.join(model_path, f'bestscore_{best_score}.hdf5'))
        else:
            if patience == config.patience:
                print(f'Early Stopping at {epoch}, score is {score}')
                break
            patience += 1



if __name__=='__main__':
    import sys
    main(get_param(sys.argv[1:]))