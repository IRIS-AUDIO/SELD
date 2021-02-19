import tensorflow as tf
import pdb
def get_MMSE(class_num):
    def MMSE(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        sed = tf.reshape(y_true, (*y_true.shape[:-1], 3, -1))
        sed = tf.reduce_sum(sed ** 2, axis=-2)

        # rest is similar
        sed = tf.keras.backend.repeat_elements(sed, 3, -1)
        sed = tf.keras.backend.cast(sed, y_pred.dtype)
        
        return tf.keras.backend.sqrt(
            tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred) * sed)) \
                    / tf.keras.backend.sum(sed)
    return MMSE

if __name__ == '__main__':
    mms = get_MMSE(14)
    pdb.set_trace()