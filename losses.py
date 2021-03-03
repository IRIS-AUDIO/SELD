import tensorflow as tf


def MMSE(y_true, y_pred):
    ''' Masked MSE '''
    y_true = tf.cast(y_true, y_pred.dtype)
    sed = tf.reshape(y_true, (*y_true.shape[:-1], 3, -1))
    sed = tf.round(tf.reduce_sum(sed ** 2, axis=-2))

    sed = tf.concat([sed] * 3, axis=-1)

    return tf.keras.backend.sqrt(
        tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred) * sed)) \
                / tf.keras.backend.sum(sed)
