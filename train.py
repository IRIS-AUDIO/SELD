import tensorflow as tf
from params import get_param
from models import build_seldnet
import pdb

@tf.function
def trainstep(model, x, y, sed_loss, doa_loss, config, optimizer):
    with tf.GradientTape() as tape:
        y_p = model(x, training=True)
        sloss = sed_loss(y[0], y_p[0])
        dloss = doa_loss(y[1], y_p[1])
        loss = sloss * int(config.loss_weight.split(',')[0]) + dloss * int(config.loss_weight.split(',')[1])
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainalbe_variables))

    return sloss, dloss

@tf.function
def teststep(model, x, y, sed_loss, doa_loss, config):
    y_p = model(x, training=False)
    sloss = sed_loss(y[0], y_p[0])
    dloss = doa_loss(y[1], y_p[1])
    return sloss, dloss


def iterloop(model, dataset, sed_loss, doa_loss, config, optimizer=None, train=True):
    if train:
        for x, y in dataset:
            sloss, dloss = trainstep(model, x, y, sed_loss, doa_loss, config, optimizer)
            
    else:
        for x, y in dataset:
            sloss, dloss = teststep(model, x, y, sed_loss, doa_loss, config)




def main(config):

    # data load
    x = tf.random.uniform((config.batch, 127, 300, 3))# [batch, freq, time, chan]
    y_1 = tf.random.uniform((config.batch, 300, 14))
    y_2 = tf.random.uniform((config.batch, 300, 14*3))
    trainset = tf.data.Dataset.from_tensor_slices((x, (y_1,y_2)))

    # data preprocess

    # model load
    model = build_seldnet((config.batch, 127, 300, 3))

    optimizer = tf.keras.optimizers.Adam()
    sed_loss = tf.keras.losses.BinaryCrossentropy(name='sed_loss')
    doa_loss = tf.keras.losses.MeanSquaredError(name='doa_loss')
    
    # metric
    total_loss = tf.keras.metrics.Mean(name='total_loss')

    for epoch in range(config.epoch):
        # train loop
        iterloop(model, trainset, sed_loss, doa_loss, config, optimizer=optimizer, train=True) 

        # validation loop


        # evaluation loop


        # tensorboard

if __name__=='__main__':
    import sys
    main(get_param(sys.argv[1:]))