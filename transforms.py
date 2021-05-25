import tensorflow as tf
from numpy import pi
from utils import safe_div


def mask(specs, axis, max_mask_size=None, period=None, n_mask=1):
    if period is None:
        period = tf.shape(specs)[axis]
    total = tf.shape(specs[:period])[axis]
    if max_mask_size is None:
        max_mask_size = total
        
    def _mask(specs):
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
    if shape[0] % period != 0:
        raise ValueError('(spec time length / period)\' rest must be 0')
    specs = tf.signal.frame(specs, period, period, axis=0)
    specs = tf.map_fn(_mask, specs)
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


def tf_cond(x):
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]
    # Replace NaNs in r with infinite unless there were NaNs before
    x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
    r_nan = tf.math.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))
    r = tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
    return r


def is_invertible(x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64
    # reference: https://stackoverflow.com/questions/57073381/how-to-check-if-a-matrix-is-invertible-in-tensorflow
    eps_inv = tf.cast(1 / epsilon, x.dtype)
    x_cond = tf_cond(x)
    return tf.math.is_finite(x_cond) & (x_cond < eps_inv)


def stab(matrix, num_channel, theta):
    # matrix: (batch, freq, chan, chan)
    nx = tf.newaxis
    dd = tf.constant([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=matrix.dtype)
    for i in range(6):
        mask = 1 - tf.cast(is_invertible(matrix), matrix.dtype)
        matrix = matrix + (mask * dd[i])[..., nx, nx] * tf.eye(matrix.shape[-1], dtype=matrix.dtype)[nx, nx,...]

    return matrix


def mcs_aug(iteration: int, theta = 1e-6):
    # reference: https://github.com/funcwj/cgmm-mask-estimator.git
    def _mcs_aug(x, y):
        '''
            x: (batch, time, freq, chan)
            y: (batch, time, n_classes*4)
        ''' 
        batch, time, freq, chan = x.shape
        x = tf.cast(x, tf.float64) # because of overflow and underflow

        # initialize rnoisy, rnoise, phi's
        rnoisy = tf.matmul(tf.transpose(x, [0, 2, 3, 1]), tf.transpose(x, [0, 2, 1, 3])) / time # (batch, freq, chan, chan)
        rnoise =  tf.tile(tf.eye(chan, dtype=x.dtype)[tf.newaxis, tf.newaxis, ...], tf.TensorShape([1,freq,1,1])) # (batch, freq, chan, chan)

        yx = x[..., tf.newaxis] # (batch, time, freq, chan, 1)
        yyh = tf.matmul(yx, tf.transpose(yx, [0, 1, 2, 4, 3])) # (batch, time, freq, chan, chan)
        rnoisy_onbin = stab(rnoisy, chan, theta)
        rnoise_onbin = stab(rnoise, chan, theta)
        
        rnoisy_inv = tf.linalg.inv(rnoisy_onbin) # (batch, freq, chan, chan)
        rnoise_inv = tf.linalg.inv(rnoise_onbin) # (batch, freq, chan, chan)

        phi_noisy = tf.math.real(tf.linalg.trace(tf.matmul(yyh, rnoisy_inv[:,tf.newaxis,...]) / chan)) # (batch, time, freq)
        phi_noise = tf.math.real(tf.linalg.trace(tf.matmul(yyh, rnoise_inv[:,tf.newaxis,...]) / chan)) # (batch, time, freq)
        # --------------------------------initialize end--------------------------------
        

        for it in range(iteration):
            rnoisy_onbin = stab(rnoisy, chan, theta)
            rnoise_onbin = stab(rnoise, chan, theta)

            rnoisy_inv = tf.linalg.inv(rnoisy_onbin)
            rnoise_inv = tf.linalg.inv(rnoise_onbin)

            # corre = yyh
            k_noise = tf.matmul(x[...,tf.newaxis,:], safe_div(rnoise_inv[:,tf.newaxis,...], phi_noise[...,tf.newaxis,tf.newaxis]))
            k_noise = tf.squeeze(tf.matmul(k_noise, x[...,tf.newaxis]), axis=(-2,-1))
            det_noise =  tf.linalg.det(phi_noise[...,tf.newaxis,tf.newaxis] * rnoise_onbin[:,tf.newaxis]) * pi
            p_noise = tf.math.real(safe_div(tf.math.exp(-k_noise), det_noise)) + theta

            k_noisy = tf.matmul(x[...,tf.newaxis,:], safe_div(rnoisy_inv[:,tf.newaxis,...], phi_noisy[...,tf.newaxis,tf.newaxis]))
            k_noisy = tf.squeeze(tf.matmul(k_noisy, x[...,tf.newaxis]), axis=(-2,-1))
            det_noisy =  tf.linalg.det(phi_noisy[...,tf.newaxis,tf.newaxis] * rnoisy_onbin[:,tf.newaxis]) * pi
            p_noisy = tf.math.real(safe_div(tf.math.exp(-k_noisy), det_noisy)) + theta

            lambda_noise = safe_div(p_noise, (p_noise + p_noisy))
            lambda_noisy = safe_div(p_noisy, (p_noise + p_noisy))

            phi_noise = tf.math.real(tf.linalg.trace(tf.matmul(yyh, rnoise_inv[:,tf.newaxis,...]) / chan))
            phi_noisy = tf.math.real(tf.linalg.trace(tf.matmul(yyh, rnoisy_inv[:,tf.newaxis,...]) / chan))

            rnoisy_accu = safe_div(lambda_noisy, phi_noisy)[...,tf.newaxis,tf.newaxis] * yyh
            rnoise_accu = safe_div(lambda_noise, phi_noise)[...,tf.newaxis,tf.newaxis] * yyh
            
            rnoisy = safe_div(tf.reduce_sum(rnoisy_accu, axis=1) , tf.reduce_sum(lambda_noisy, axis=1)[...,tf.newaxis, tf.newaxis])
            rnoise = safe_div(tf.reduce_sum(rnoise_accu, axis=1) , tf.reduce_sum(lambda_noise, axis=1)[...,tf.newaxis, tf.newaxis])
        
        x = tf.cast(x, tf.float32) * tf.cast(lambda_noise[...,tf.newaxis], tf.float32)

        return x, y 
    return _mcs_aug

