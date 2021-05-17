import copy
import tensorflow as tf

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


def dict_add(first: dict, second: dict):
    output = copy.deepcopy(first)

    for key in second.keys():
        if key in output:
            output[key] += second[key]
        else:
            output[key] = second[key]

    return output


def safe_tuple(tuple_or_scalar, length=2):
    if isinstance(tuple_or_scalar, (int, float)):
        tuple_or_scalar = (tuple_or_scalar, ) * length

    tuple_or_scalar = tuple(tuple_or_scalar)
    count = len(tuple_or_scalar)
    if count == 1:
        tuple_or_scalar = tuple_or_scalar * length
    elif count != length:
        raise ValueError("length of input must be one or required length")
    return tuple_or_scalar


def force_1d_shape(shape):
    # shape must not have batch dim
    if len(shape) == 3:
        shape = [shape[0], shape[1] * shape[2]]
    elif len(shape) > 3:
        raise ValueError(f'invalid shape: {shape}')
    return shape


def get_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads
    