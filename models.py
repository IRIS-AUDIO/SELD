import tensorflow as tf
from tensorflow.keras.layers import *


def build_seldnet(data_in, 
                  n_classes=1, 
                  dropout_rate=0., 
                  nb_cnn2d_filt=64, 
                  pool_size=None,
                  rnn_size=None, 
                  fnn_size=None):
    '''
    regression SELDnet
    data_in: [batch, freq, time, chan]
    '''
    if pool_size is None:
        pool_size = [8, 8, 2]
    if rnn_size is None:
        rnn_size = [128, 128]
    if fnn_size is None:
        fnn_size = [128]

    # model definition
    spec_start = Input(shape=data_in[-3:])
    spec_cnn = spec_start

    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(nb_cnn2d_filt, kernel_size=3, padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(pool_size[i], 1))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    # [b, f, t, c] -> [b, t, c]
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    spec_rnn = Reshape((-1, spec_cnn.shape[-2]*spec_cnn.shape[-1]))(spec_cnn)

    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', 
                dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                return_sequences=True),
            merge_mode='mul')(spec_rnn)

    # SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = Dense(nb_fnn_filt)(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)

    # DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = Dense(nb_fnn_filt)(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = Dense(n_classes*3, activation='tanh', name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

