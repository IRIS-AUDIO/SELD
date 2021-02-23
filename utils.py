import tensorflow as tf
import torch

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, decay):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.decay = decay

    def __call__(self, epoch):

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def safe_div(x, y, eps=1e-8):
    # returns safe x / max(y, epsilon)
    return x / tf.maximum(y, eps)


def get_device():
    return torch.device('cuda:' if torch.cuda.is_available() else 'cpu')