import tensorflow as tf
from tensorflow.keras.layers import *


def initial_seldnet(data_in, 
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
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(n_classes, activation='sigmoid', name='sed_out'))(sed)

    # DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = TimeDistributed(Dense(n_classes*3, activation='tanh', name='doa_out'))(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

def initial_seldnet_v1(data_in, 
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
    doa = Dense(n_classes*3, activation='tanh')(doa)
    doa = Reshape((-1, 3, n_classes))(doa)
    doa *= Reshape((-1, 1, n_classes))(sed)
    doa = Reshape((-1, 3*n_classes), name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

def initial_seldnet_v2(data_in, 
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
    seddoa = Dense(n_classes * 3, activation='sigmoid')(sed)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(seddoa)

    # DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = Dense(nb_fnn_filt)(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = Dense(n_classes*3, activation='tanh')(doa)
    doa = Reshape((-1, 3, n_classes))(doa)
    doa *= Reshape((-1, 3, n_classes))(seddoa)
    doa = Reshape((-1, 3*n_classes), name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

def initial_seldnet_v3(data_in, 
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
    weight = Dense(3, activation='sigmoid')(sed)[..., tf.newaxis]
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)
    
    weightedsed = weight * Reshape((sed.shape[-2], 1, -1))(sed)

    # DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = Dense(nb_fnn_filt)(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = Dense(n_classes*3, activation='tanh')(doa)
    doa = Reshape((-1, 3, n_classes))(doa)
    doa *= Reshape((-1, 3, n_classes))(weightedsed)
    doa = Reshape((-1, 3*n_classes), name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model


def attention_seldnet(data_in, 
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
    spec_att = Reshape((-1, spec_cnn.shape[-2]*spec_cnn.shape[-1]))(spec_cnn)

    # Attention
    nb_rnn_filt = 128
    # for nb_rnn_filt in rnn_size:
    H, m, c = LSTM(nb_rnn_filt, return_sequences=True, return_state=True)(spec_att)
    S, h, c = LSTM(nb_rnn_filt, return_sequences=True, return_state=True)(spec_att, initial_state=[m,c])
    S_ = tf.concat([m[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
    A = Attention()([S_, H])
    y = Concatenate(-1)([S, A])
    spec_att = Dense(nb_rnn_filt)(y)

    # SED
    sed = spec_att
    for nb_fnn_filt in fnn_size:
        sed = Dense(nb_fnn_filt)(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)

    # DOA
    doa = spec_att
    for nb_fnn_filt in fnn_size:
        doa = Dense(nb_fnn_filt)(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = Dense(n_classes*3, activation='tanh', name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

def attention_seldnet_v1(data_in, 
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
    spec_att = Reshape((-1, spec_cnn.shape[-2]*spec_cnn.shape[-1]))(spec_cnn)

    # Attention
    nb_rnn_filt = 128
    spec_att = tf.keras.layers.MultiHeadAttention(32, nb_rnn_filt)(spec_att, spec_att)

    # SED
    sed = spec_att
    for nb_fnn_filt in fnn_size:
        sed = Dense(nb_fnn_filt)(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = Dense(n_classes, activation='sigmoid', name='sed_out')(sed)

    # DOA
    doa = spec_att
    for nb_fnn_filt in fnn_size:
        doa = Dense(nb_fnn_filt)(doa)
        doa = Dropout(dropout_rate)(doa)
    doa = Dense(n_classes*3, activation='tanh', name='doa_out')(doa) 

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model
