import tensorflow as tf


def MMSE(y_true, y_pred):
    ''' Masked MSE '''
    y_true = tf.cast(y_true, y_pred.dtype)
    sed = tf.reshape(y_true, (*y_true.shape[:-1], 3, -1))
    sed = tf.round(tf.reduce_sum(sed ** 2, axis=-2))

    sed = tf.concat([sed] * 3, axis=-1)

    return tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred) * sed) \
            / tf.keras.backend.sum(sed)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    focal = - y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)\
            - (1-y_true) * alpha * tf.pow(y_pred, gamma) * tf.math.log(1-y_pred)
    return tf.reduce_mean(focal)


class Focal_Loss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
        focal = - y_true * self.alpha * tf.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)\
                - (1-y_true) * self.alpha * tf.pow(y_pred, self.gamma) * tf.math.log(1-y_pred)
        return tf.reduce_mean(focal)
 
