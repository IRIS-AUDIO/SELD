import tensorflow as tf


def mask(specs, axis, max_mask_size=None, n_mask=1):
    def make_shape(size):
        # returns (1, ..., size, ..., 1)
        shape = [1] * len(specs.shape)
        shape[axis] = size
        return tuple(shape)

    total = tf.shape(specs)[axis]
    mask = tf.ones(make_shape(total), dtype=specs.dtype)
    if max_mask_size is None:
        max_mask_size = total

    def apply_random_mask(mask):
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

        mask *= tf.concat(
            (tf.ones(shape=make_shape(offset), dtype=mask.dtype),
             tf.zeros(shape=make_shape(size), dtype=mask.dtype),
             tf.ones(shape=make_shape(total-size-offset), dtype=mask.dtype)),
            axis=axis)
        return mask

    i = tf.constant(0)
    cond = lambda i, m: i < n_mask
    body = lambda i, m: (i+1, apply_random_mask(m))
    _, mask = tf.while_loop(cond, body, (i, mask))

    return specs * mask


def foa_intensity_vec_aug(x, y):
    # x : [batch, time, freq, 7]
    # y : [batch, time, 4*n_classes]
    x = tf.identity(x)
    y = tf.identity(y)
    # [batch, time, 4*n_classes] to [batch, time, 4, n_classes]
    y = tf.reshape(y, [*y.shape[:-1]] + [4, -1]) 

    intensity_vectors = x[..., -3:]
    cartesian = y[..., -3:, :]

    batch = x.shape[0]
    flip = tf.random.uniform([batch, 3], 0, 2, dtype=tf.int32)
    flip = tf.cast(flip, 'float32')

    intensity_vectors = (1 - 2*tf.reshape(flip, (-1, 1, 1, 3))) * intensity_vectors 
    cartesian = (1 - 2*tf.reshape(flip, (-1, 1, 3, 1))) * cartesian 

    x = tf.concat([x[..., :-3], intensity_vectors], axis=-1)
    y = tf.concat([y[..., :-3, :], cartesian], axis=-2)
    y = tf.reshape(y, [*y.shape[:-2]] + [4*y.shape[-1]])

    return x, y


def split_total_labels_to_sed_doa(x, y):
    n_classes = tf.shape(y)[-1] // 4
    return x, (y[..., :n_classes], y[..., n_classes:])

