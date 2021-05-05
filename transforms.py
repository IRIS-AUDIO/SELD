import tensorflow as tf


def mask(specs, axis, max_mask_size=None, period=100, n_mask=1):
    total = tf.shape(specs[:period])[axis]
    if max_mask_size is None:
        max_mask_size = total
        
    def _mask(inputs):
        specs, max_mask_size = inputs[0], inputs[1]
        def make_shape(size):
            # returns (1, ..., size, ..., 1)
            shape = [1] * len(specs.shape)
            shape[axis] = size
            return tuple(shape)

        mask = tf.ones(make_shape(total), dtype=specs.dtype)

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

    shape = tf.shape(specs)
    specs = tf.signal.frame(specs, period, period, axis=0)
    max_mask_size = tf.repeat(max_mask_size, tf.shape(specs)[0])
    specs = tf.map_fn(_mask, (specs, max_mask_size), dtype=(tf.float32, tf.int32), fn_output_signature=tf.float32)
    specs = tf.reshape(specs, shape)
    return specs

def foa_intensity_vec_aug(x, y):
    # x : [batch, time, freq, 7]
    # y : [batch, time, 4*n_classes]
    x = tf.identity(x)
    y = tf.identity(y)
    batch_size = tf.shape(x)[0]
    # [batch, time, 4*n_classes] to [batch, time, 4, n_classes]
    y = tf.reshape(y, [-1] + [*y.shape[1:-1]] + [4, y.shape[-1]//4])

    intensity_vectors = x[..., -3:]
    cartesian = y[..., -3:, :]

    flip = tf.random.uniform([batch_size, 3], 0, 2, dtype=tf.int32)
    flip = tf.cast(flip, 'float32')

    intensity_vectors = (1 - 2*tf.reshape(flip, (-1, 1, 1, 3))) * intensity_vectors 
    cartesian = (1 - 2*tf.reshape(flip, (-1, 1, 3, 1))) * cartesian

    correct_shape = tf.tile([[0,1,2]], [batch_size, 1])
    
    # x,y축 회전
    perm = 2 * tf.random.uniform([batch_size, 1], maxval=2, dtype=tf.int32)
    perm = tf.concat([perm, tf.ones_like(perm), 2-perm], axis=-1)

    # x,y,z축 회전
    # perm = tf.map_fn(tf.random.shuffle, correct_shape)
    
    check = tf.reduce_sum(tf.cast(perm != correct_shape, tf.int32), -1, keepdims=True)
    feat_perm = (perm + check) % 3

    intensity_vectors = tf.gather(intensity_vectors, feat_perm, axis=-1, batch_dims=1)
    cartesian = tf.gather(cartesian, feat_perm, axis=-2, batch_dims=1)
    
    x = tf.concat([x[..., :1], tf.gather(x[..., 1:4], perm, axis=-1, batch_dims=1), intensity_vectors], axis=-1)
    
    y = tf.concat([y[..., :-3, :], cartesian], axis=-2)
    y = tf.reshape(y, [-1] + [*y.shape[1:-2]] + [4*y.shape[-1]])

    return x, y


def split_total_labels_to_sed_doa(x, y):
    n_classes = tf.shape(y)[-1] // 4
    return x, (y[..., :n_classes], y[..., n_classes:])


def mic_gcc_perm(mic_perm):
    '''
        notice:
            This function is only available in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] ordered gcc feature

        inputs:
            mic_perm: [batch_size, 4] perm dimension number

        outputs:
            gcc_perm: [batch_size, 6] gcc perm dimension number
    '''
    batch_size = tf.shape(mic_perm)[0]
    current_gcc_dim = tf.tile([[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]], [batch_size, 1, 1])
    decode_table = tf.constant([[0,0,1,2],[0,0,3,4],[1,3,0,5],[2,4,5,0]], dtype=tf.int32)
    res = tf.gather_nd(mic_perm - tf.range(4, dtype=mic_perm.dtype)[tf.newaxis, ...], current_gcc_dim[...,tf.newaxis], batch_dims=1) + current_gcc_dim
    gcc_perm = tf.gather_nd(decode_table, res)
    return gcc_perm


# reference: https://arxiv.org/pdf/2101.02919.pdf, TABLE 1
# [[mic channel], [foa channel]]
channel_list = [
    [[1,3,0,2], [0,-3,-2,1]],
    [[3,1,2,0], [0,-3,2,-1]],
    [[0,1,2,3], [0,1,2,3]],
    [[1,0,3,2], [0,-1,-2,3]],
    [[2,0,3,1], [0,3,-2,-1]],
    [[0,2,1,3], [0,3,2,1]],
    [[3,2,1,0], [0,-1,2,-3]],
    [[2,3,0,1], [0,1,-2,-3]]
]


def acs_aug(x, y):
    '''
        acs: Audio Channel Swapping
    '''
    # x : [batch, time, freq, 17],
    # 4: foa, 3: intensity vector, 4: mic, 6: gcc-phat
    # y : [batch, time, 4*n_classes]
    x = tf.identity(x)
    y = tf.identity(y)
    batch_size = tf.shape(x)[0]
    # [batch, time, 4*n_classes] to [batch, time, 4, n_classes]
    y = tf.reshape(y, [-1] + [*y.shape[1:-1]] + [4, y.shape[-1]//4])

    # foa
    intensity_vectors = x[..., 4:7]
    cartesian = y[..., -3:, :]

    correct_shape = [0,1,2]
    idx = tf.random.uniform([batch_size], 0, 8, dtype=tf.int32)
    flip = tf.gather(channel_list, idx)
    foa_flip = flip[...,1,1:]

    foa_sign = tf.sign(foa_flip)
    foa_perm = foa_sign * foa_flip - 1
    foa_sign = tf.cast(foa_sign, intensity_vectors.dtype)
    check = tf.reduce_sum(tf.cast(foa_perm != correct_shape, tf.int32), -1, keepdims=True)
    foa_feat_perm = (foa_perm + check) % 3
    foa_x = tf.gather(x[..., 1:4], foa_perm, axis=-1, batch_dims=1)

    intensity_vectors = tf.gather(intensity_vectors, foa_feat_perm, axis=-1, batch_dims=1) * foa_sign[:,tf.newaxis,tf.newaxis,:]
    cartesian = tf.gather(cartesian, foa_feat_perm, axis=-2, batch_dims=1) * foa_sign[:,tf.newaxis,:,tf.newaxis]

    # mic
    mic_flip = flip[...,0,:]
    gcc_phat = x[..., 11:]
    gcc_perm = mic_gcc_perm(mic_flip)
    gcc_phat = tf.gather(gcc_phat, gcc_perm, axis=-1, batch_dims=1)
    mic_x = tf.gather(x[..., 7:11], mic_flip, axis=-1, batch_dims=1)

    x = tf.concat([x[..., :1], foa_x, intensity_vectors, mic_x, gcc_phat], axis=-1)
    
    y = tf.concat([y[..., :-3, :], cartesian], axis=-2)
    y = tf.reshape(y, [-1] + [*y.shape[1:-2]] + [4*y.shape[-1]])

    return x, y

