import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from layers import *


def seldnet(data_in,
            hlfr,
            tcr,
            sedl,
            doal,
            n_classes=14):
    '''
    regression SELDnet
    data_in: [batch, time, freq, chan]
    hlfr: high-level feature representation
    tcr: temporal context representation
    sedl: sed-layer
    doal: doa-layer
    '''
    # model definition
    spec_start = Input(shape=data_in[-3:])

    x = hlfr(spec_start)

    x = tcr(x)

    # sed
    sed = sedl(x)
    sed = sigmoid(sed)

    # doa
    doa = doal(x)
    doa = tanh(doa)

    return tf.keras.Model(inputs=spec_start, outputs=[sed, doa])


def seldnet_v1(data_in, 
                  n_classes=1, 
                  dropout_rate=0., 
                  nb_cnn2d_filt=64, 
                  pool_size=None,
                  rnn_size=None, 
                  fnn_size=None):
    '''
    regression SELDnet
    data_in: [batch, time, freq, chan]
    hlfr: high-level feature representation
    tcr: temporal context representation
    sedl: sed-layer
    doal: doa-layer
    '''
    
    x = hlfr(spec_start)

    x = tcr(x)

    # sed
    sed = sedl(x)
    sed = sigmoid(sed)

    # doa
    doa = doal(x)
    doa *= Reshape((-1, 1, n_classes))(sed)
    doa = Reshape((-1, 3*n_classes), name='doa_out')(doa)
    doa = tanh(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model


def seldnet_architecture(input_shape, hlfr, tcr, sed, doa):
    '''
    regression SELDnet
    input_shape: [batch, time, freq, chan]
    hlfr: high-level feature representation
    tcr: temporal context representation
    sed: sed-layer
    doa: doa-layer
    '''
    inputs = Input(shape=input_shape)

    x = hlfr(inputs)
    x = tcr(x)

    # sed
    sed_out = sed(x)
    sed_out = sigmoid(sed_out)

    # doa
    doa_out = doa(x)
    doa_out = tanh(doa_out)

    return tf.keras.Model(inputs=inputs, outputs=[sed_out, doa_out])

