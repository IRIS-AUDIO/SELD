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
    y = tf.reshape(y, [-1] + [*y.shape[1:-1]] + [4, y.shape[-1]//4])

    intensity_vectors = x[..., -3:]
    cartesian = y[..., -3:, :]

    flip = tf.random.uniform([tf.shape(x)[0], 3], 0, 2, dtype=tf.int32)
    flip = tf.cast(flip, 'float32')

    intensity_vectors = (1 - 2*tf.reshape(flip, (-1, 1, 1, 3))) * intensity_vectors 
    cartesian = (1 - 2*tf.reshape(flip, (-1, 1, 3, 1))) * cartesian 

    x = tf.concat([x[..., :-3], intensity_vectors], axis=-1)
    y = tf.concat([y[..., :-3, :], cartesian], axis=-2)
    y = tf.reshape(y, [-1] + [*y.shape[1:-2]] + [4*y.shape[-1]])

    return x, y


def split_total_labels_to_sed_doa(x, y):
    n_classes = tf.shape(y)[-1] // 4
    return x, (y[..., :n_classes], y[..., n_classes:])


''' For FOA Channel Swapping '''
T_XYZ_TO_YZX = tf.convert_to_tensor(
    [[0., 1., 0.],
     [0., 0., 1.],
     [1., 0., 0.]])

T_XYZ_TO_YZX_INV = tf.convert_to_tensor(
    [[0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.]])

I3 = tf.eye(3, dtype='float32')

def get_xyz_swap(yzx_swap):
    B = tf.gather(I3, yzx_swap)
    A = tf.matmul(tf.matmul(T_XYZ_TO_YZX_INV, B), T_XYZ_TO_YZX)
    return tf.argmax(A, -1)

