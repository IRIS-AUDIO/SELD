import copy
import tensorflow as tf
import numpy as np 
import csv
import os
import pandas as pd

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
    

class AdaBelief(tf.keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='AdaBelief',
                 **kwargs):
        super(AdaBelief, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBelief, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
                    (tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdaBelief, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = tf.compat.v1.assign(m, 
                               m * coefficients['beta_1_t'] + m_scaled_g_values,
                               use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * ((g_t-m_t) * (g_t-m_t))
        v = self.get_slot(var, 'v')
        grad_dev = grad - m_t 
        v_scaled_g_values = (grad_dev * grad_dev) * coefficients['one_minus_beta_2_t']
        v_t = tf.compat.v1.assign(v, 
                                  v * coefficients['beta_2_t'] + v_scaled_g_values,
                                  use_locking=self._use_locking)

        if not self.amsgrad:
            v_sqrt = tf.math.sqrt(v_t)
            var_update = tf.compat.v1.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = tf.math.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = tf.compat.v1.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = tf.math.sqrt(v_hat_t)
            var_update = tf.compat.v1.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t, v_hat_t])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = tf.compat.v1.assign(m, m * coefficients['beta_1_t'],
                                                     use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * ((g_t-m_t) * (g_t-m_t))
        v = self.get_slot(var, 'v')
        grad_dev = grad - m_t 
        v_scaled_g_values = (grad_dev * grad_dev) * coefficients['one_minus_beta_2_t']
        v_t = tf.compat.v1.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = tf.math.sqrt(v_t)
            var_update = tf.compat.v1.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = tf.math.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = tf.compat.v1.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = tf.math.sqrt(v_hat_t)
            var_update = tf.compat.v1.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(AdaBelief, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config

def write_answer(output_dir, filename, preds, direction):
    write_path = os.path.join(output_dir, filename)
    loc_answer = tf.where(preds)
    true_direction = []
    for loc in loc_answer:
        true_direction.append(direction[loc[0], loc[1]::preds.shape[1]])
    true_direction = tf.stack(true_direction, axis=0)
    true_direction = true_direction.numpy()

    #azimuth = np.arctan2(true_direction[:,1], true_direction[:,0]) * 180 / np.pi
    #elevation = np.arctan2(true_direction[:,2], np.sqrt(true_direction[:,0]**2 + true_direction[:,1]**2)) * 180 / np.pi
    
    #azimuth = azimuth.reshape(azimuth.shape[0], 1)
    #elevation = elevation.reshape(elevation.shape[0], 1)

    temp = np.concatenate([loc_answer.numpy(), true_direction.reshape(true_direction.shape[0], 3)], axis=1)
    # np.savetxt(write_path, temp.astype(float), fmt='%4.3f', delimiter = ",")
    _fid = open(write_path, 'w')
    for item in temp:
        _fid.write('{},{},{},{},{},{}\n'.format(int(item[0]), int(item[1]), 0, float(item[2]), float(item[3]), float(item[4])))


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(float(_words[0]))
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #read polar coordinates format, we ignore the track count 
            _output_dict[_frame_ind].append([int(float(_words[1])), float(_words[3]), float(_words[4]), int(float(_words[2]))])
        elif len(_words) == 6: # read Cartesian coordinates format, we ignore the track count
            _output_dict[_frame_ind].append([int(float(_words[1])), float(_words[3]), float(_words[4]), float(_words[5]), int(float(_words[2]))])
    _fid.close()
    return _output_dict

def segment_labels( _pred_dict, _max_frames):
    nb_blocks = int(np.ceil(_max_frames/float(10)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, 10):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // 10
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+10):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}
                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation, tmp_val[-1]])
    return out_dict


def apply_kernel_regularizer(model, kernel_regularizer):
    model = tf.keras.models.clone_model(model)
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer

    model = tf.keras.models.clone_model(model)
    return model

def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[2]*np.pi/180.
                azi_rad = tmp_val[1]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], x, y, z, tmp_val[-1]])
    return out_dict