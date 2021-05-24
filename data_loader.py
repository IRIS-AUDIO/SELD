import numpy as np
import tensorflow as tf
from feature_extractor import *
import random as rnd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
import tensorflow_io as tfio
import joblib
AUTOTUNE = tf.data.experimental.AUTOTUNE


def data_loader(dataset, 
                preprocessing=None,
                sample_transforms=None, 
                batch_transforms=None,
                deterministic=False,
                loop_time=None,
                batch_size=32) -> tf.data.Dataset:
    '''
    INPUT
        preprocessing: a list of preprocessing ops
                       output of preprocessing ops will be cached
        sample_transforms: a list of samplewise augmentations
        batch_transforms: a list of batchwise augmentations
        deterministic: set to False for efficiency,
                       if the order of the data is critical, set to True
        inf_loop: whether to loop infinitely (will run .repeat() after .cache())
                  this can also increase efficiency
        batch_size: batch size
    '''
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

    def apply_ops(dataset, operations):
        if operations is None:
            return dataset

        if not isinstance(operations, (list, tuple)):
            operations = [operations]

        for op in operations:
            dataset = dataset.map(
                op, num_parallel_calls=AUTOTUNE, deterministic=deterministic)

        return dataset

    dataset = apply_ops(dataset, preprocessing)
    dataset = dataset.cache()
    dataset = dataset.repeat(loop_time)
    dataset = apply_ops(dataset, sample_transforms)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = apply_ops(dataset, batch_transforms)

    return dataset


def load_seldnet_data(feat_path, label_path, mode='train', n_freq_bins=64):
    from glob import glob
    import os

    assert mode in ['train', 'val', 'test']
    splits = {
        'train': [3, 4, 5, 6],
        'val': [2],
        'test': [1]
    }

    # load splits according to the mode
    if not os.path.exists(feat_path):
        raise ValueError(f'no such feat_path ({feat_path}) exists')
    features = sorted(glob(os.path.join(feat_path, '*.npy')))
    features = [np.load(f).astype('float32') for f in features 
                if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if not os.path.exists(label_path):
        raise ValueError(f'no such label_path ({label_path}) exists')
    labels = sorted(glob(os.path.join(label_path, '*.npy')))
    labels = [np.load(f).astype('float32') for f in labels
              if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if len(features[0].shape) == 2:
        def extract(x):
            x_org = x[:, :n_freq_bins*4]
            x_org = np.reshape(x_org, (x.shape[0], n_freq_bins, 4))
            x_add = x[:, n_freq_bins*4:]
            x_add = np.reshape(x_add, (x.shape[0], n_freq_bins, -1))
            return np.concatenate([x_org, x_add], axis=-1)

        features = list(map(extract, features))
    else:
        # already in shape of [time, freq, chan]
        pass
    
    return features, labels


def load_wav_and_label(feat_path, label_path, mode='train'):
    '''
        output
        x: wave form -> (data_num, channel(4), time)
        y: label(padded) -> (data_num, time, 56)
    '''
    f_paths = sorted(glob(os.path.join(feat_path, '*.wav')))
    l_paths = sorted(glob(os.path.join(label_path, '*.csv')))

    splits = {
        'train': [3, 4, 5, 6],
        'val': [2],
        'test': [1]
    }

    f_paths = [f for f in f_paths 
            if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]
    l_paths = [f for f in l_paths 
            if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    if len(f_paths) != len(l_paths):
        raise ValueError('# of features and labels are not matched')
    
    def preprocess_label(labels, max_label_length=600):
        cur_len = labels.shape[0]
        max_len = max_label_length

        if cur_len < max_len: 
            labels = tf.pad(labels, ((0, max_len-cur_len), (0,0)))
        else:
            labels = labels[:max_len]
        return labels
    x = list(map(lambda x: tf.transpose(tf.audio.decode_wav(tf.io.read_file(x))[0]), f_paths))
    y = list(map(lambda x: preprocess_label(extract_labels(x)), l_paths))
    return x, y


def seldnet_data_to_dataloader(features: [list, tuple], 
                               labels: [list, tuple], 
                               train=True, 
                               label_window_size=60,
                               drop_remainder=True,
                               shuffle_size=None,
                               batch_size=32,
                               loop_time=1,
                               **kwargs):
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # shapes of seldnet features and labels 
    # features: [time_features, freq, chan]
    # labels:   [time_labels, 4*classes]
    # for each 5 input time slices, a single label time slices was designated
    # features' shape [time_f, freq, chan] -> [time_l, resolution, freq, chan]
    features = np.reshape(features, (labels.shape[0], -1, *features.shape[1:]))

    # windowing
    n_samples = features.shape[0] // label_window_size
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(label_window_size, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda x,y: (tf.reshape(x, (-1, *x.shape[2:])), y),
                          num_parallel_calls=AUTOTUNE)
    del features, labels
    
    dataset = data_loader(dataset, batch_size=batch_size, 
            loop_time=loop_time if train else 1, **kwargs)
    
    if train:
        if shuffle_size is None:
            shuffle_size = n_samples // batch_size
        dataset = dataset.shuffle(shuffle_size)

    return dataset.prefetch(AUTOTUNE)


def get_TDMset(TDM_PATH):
    from glob import glob
    tdm_path = os.path.join(TDM_PATH, 'foa_dev_tdm')
    class_num = len(glob(tdm_path + '/*label_*.joblib'))

    def load_data(cls):
        return tf.convert_to_tensor(joblib.load(os.path.join(tdm_path, f'tdm_noise_{cls}.joblib')), dtype=tf.float32)

    def load_label(cls):
        return tf.convert_to_tensor(joblib.load(os.path.join(tdm_path, f'tdm_label_{cls}.joblib')))
    
    with ThreadPoolExecutor() as pool:
        tdm_x = list(pool.map(load_data, range(class_num)))
        tdm_y = list(pool.map(load_label, range(class_num)))
    return tdm_x, tdm_y


@tf.function
def get_mixed_data(x, y, tdm_x, tdm_y, offset, sample_time, max_overlap_per_frame=2):
    class_num = tdm_y.shape[-1] // 4
    sr = tdm_x.shape[-1] // tdm_y.shape[0]
    cls = tf.reduce_max(tf.argmax(tdm_y[...,:class_num], -1))

    nondup_class = 1 - y[offset:offset+sample_time][..., cls]

    valid_index = tf.cast(tf.reduce_sum(y[offset:offset+sample_time][..., :class_num], -1) < max_overlap_per_frame, nondup_class.dtype) * nondup_class # 1프레임당 최대 클래스 개수보다 작으면서 겹치지 않는 노이즈를 넣을 수 있는 공간 찾기

    y += tdm_y * tf.pad(valid_index[...,tf.newaxis], ((offset, tdm_y.shape[0] - offset - sample_time),(0,0)))
    x += tdm_x * tf.pad(tf.repeat(valid_index, sr)[tf.newaxis,...], ((0,0),(offset * sr, (tdm_y.shape[0] - offset - sample_time) * sr)))
    return x, y


def TDM_aug(x: list, y: list, tdm_x, tdm_y, sr=24000, label_resolution=0.1, max_overlap_num=5, max_overlap_per_frame=2, min_overlap_sec=1, max_overlap_sec=5):
    '''
        x: list(tf.Tensor): shape(sample number, channel(4), frame(1440000))
        y: list(tf.Tensor): shape(sample number, time(600), class+cartesian(14+42))
        tdm_x: list(tf.Tensor): shape(class_num, channel(4), frame)
        tdm_y: list(tf.Tensor): shape(class_num, time, class+cartesian(14+42))
    '''
    class_num = y[0].shape[-1] // 4
    lab_len = y[0].shape[0]
    min_overlap_frame = int(min_overlap_sec / label_resolution) 
    max_overlap_frame = int(max_overlap_sec / label_resolution)
    sr = int(sr * label_resolution)

    stacked_y = tf.stack(y)[..., :class_num]
    weight_logit = 1 / tf.reduce_sum(stacked_y, (-3,-2))
    weight_logit /= tf.reduce_sum(weight_logit)
    
    # stacked_y = tf.stack(y)[..., :class_num]
    # total_y_frames = tf.reduce_sum(stacked_y)
    # estimated_n_frames = len(y) * max_overlap_num * (max_overlap_sec - min_overlap_sec)/2 * 0.8
    # target_n_frames_per_cls = (estimated_n_frames + total_y_frames) / class_num

    # frames_per_cls = tf.reduce_sum(stacked_y, axis=(0, 1))
    # weights = tf.nn.relu(target_n_frames_per_cls - frames_per_cls)
    # weight_logit = tf.keras.utils.normalize(weights, order=1)[0]
    
    def do_tdm(_):
        def select_random_cls(max_overlap_num):
            classes = list(map(lambda x: tf.random.categorical(tf.math.log(weight_logit[tf.newaxis,...]), max_overlap_num), range(max_overlap_num)))
            return tf.reshape(tf.concat(classes, -1), (-1,))

        selected_cls = select_random_cls(max_overlap_num)
        
        def get_tdm_frame(cls):
            cls = tf.reshape(cls, ())
            sample_time = tf.random.uniform((), min_overlap_frame, max_overlap_frame, dtype=cls.dtype) # tdmset에서 뽑을 sample의 길이
            td_offset = tf.random.uniform((), 0, tdm_y[cls].shape[0] - sample_time, dtype=cls.dtype) # tdmset에서 뽑을 sample의 offset

            using_offset = tf.random.uniform((), 0, lab_len - sample_time, dtype=cls.dtype) # 합성될 위치
            x = tdm_x[cls][..., td_offset * sr:(td_offset + sample_time) * sr]
            y = tdm_y[cls][td_offset:td_offset + sample_time]

            x = tf.pad(x, ((0,0), (using_offset * sr, (lab_len -  sample_time - using_offset) * sr)))
            y = tf.pad(y, ((using_offset, lab_len - sample_time - using_offset), (0,0)))
            return x, y, using_offset, sample_time
        
        selected_tdms = tf.map_fn(get_tdm_frame, selected_cls, 
                    fn_output_signature=(tf.TensorSpec(shape=(4, lab_len * sr), dtype=tf.float32),
                                         tf.TensorSpec(shape=(lab_len, 56), dtype=tf.float32),
                                         tf.TensorSpec(shape=(), dtype=selected_cls.dtype),
                                         tf.TensorSpec(shape=(), dtype=selected_cls.dtype)))

        i = tf.constant(0)
        cond = lambda i, j: i < max_overlap_num
        def body(i, selected_tdms):
            x[i], y[i] = get_mixed_data(x[i], y[i], selected_tdms[0][i], selected_tdms[1][i], selected_tdms[2][i], selected_tdms[3][i])
            return i + 1, selected_tdms
        tf.while_loop(cond, body, (i, selected_tdms), parallel_iterations=selected_cls.shape[0])
        return tf.zeros((), dtype=tf.int32)

    tf.map_fn(do_tdm, tf.range(len(x)))
    return x, y


def foa_intensity_vectors_tf(spectrogram, eps=1e-8):
    # complex_specs: [chan, time, freq]
    conj_zero = tf.math.conj(spectrogram[0])
    IVx = tf.math.real(conj_zero * spectrogram[3])
    IVy = tf.math.real(conj_zero * spectrogram[1])
    IVz = tf.math.real(conj_zero * spectrogram[2])

    norm = tf.math.sqrt(IVx**2 + IVy**2 + IVz**2)
    norm = tf.math.maximum(norm, eps)
    IVx = IVx / norm
    IVy = IVy / norm
    IVz = IVz / norm

    # apply mel matrix without db ...
    return tf.stack([IVx, IVy, IVz], axis=0)


def gcc_features_tf(complex_specs, n_mels):
    n_chan = complex_specs.shape[0]
    gcc_feat = []
    for m in range(n_chan):
        for n in range(m+1, n_chan):
            R = tf.math.conj(complex_specs[m]) * complex_specs[n]
            print(R.shape)
            cc = tf.signal.irfft(tf.math.exp(1.j*tf.complex(tf.math.angle(R),0.0)))
            cc = tf.concat([cc[-n_mels//2:], cc[:(n_mels+1)//2]], axis=0)
            gcc_feat.append(cc)

    return tf.stack(gcc_feat, axis=0)

def get_preprocessed_x(wav, sample_rate, mode='foa', n_mels=64,
                       multiplier=5, max_label_length=600, **kwargs):
    device = get_device()
    melscale = torchaudio.transforms.MelScale(
        n_mels=n_mels, sample_rate=sample_rate).to(device)
    spec = complex_spec(wav.to(device), **kwargs)

    mel_spec = torchaudio.functional.complex_norm(spec, power=2.)
    mel_spec = melscale(mel_spec)
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier=10.,
        amin=1e-10,
        db_multiplier=np.log10(max(1., 1e-10)), # log10(max(ref, amin))
        top_db=80.,
    )

    features = [mel_spec]
    if mode == 'foa':
        foa = foa_intensity_vectors(spec)
        foa = melscale(foa)
        features.append(foa)
    elif mode == 'mic':
        gcc = gcc_features(spec, n_mels=n_mels)
        features.append(gcc)
    else:
        raise ValueError('invalid mode')

    features = torch.cat(features, axis=0)

    # [chan, freq, time] -> [time, freq, chan]
    features = torch.transpose(features, 0, 2)
    cur_len = features.shape[0]
    max_len = max_label_length * multiplier
    if cur_len < max_len: 
        features = np.pad(features, 
                          ((0, max_len-cur_len), (0,0), (0,0)),
                          'constant')
    else:
        features = features[:max_len]

    return features

def get_preprocessed_x_tf(wav, sr, mode='foa', n_mels=64,
                          multiplier=5, max_label_length=600, win_length=1024,
                          hop_length=480, n_fft=1024):
    mel_mat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mels,
                                                    num_spectrogram_bins=n_fft//2+1,
                                                    sample_rate=sr,
                                                    lower_edge_hertz=0,
                                                    upper_edge_hertz=sr//2)


    spectrogram = tf.signal.stft(wav, win_length, hop_length, n_fft, pad_end=True)
    norm_spec = tf.math.abs(spectrogram)
    mel_spec = tf.matmul(norm_spec, mel_mat)
    mel_spec = tfio.experimental.audio.dbscale(mel_spec, top_db=80)
    features = [mel_spec]
        
    if mode == 'foa':
        foa = foa_intensity_vectors_tf(spectrogram)
        foa = tf.matmul(foa, mel_mat)
        features.append(foa)
        
    elif mode == 'mic':
        gcc = gcc_features_tf(spectrogram, n_mels=n_mels)
        features.append(gcc)
    
    else:
        raise ValueError('invalid mode')
    
    features = tf.concat(features, axis=0)
    features = tf.transpose(features, perm=[1, 2, 0])
    
    cur_len = features.shape[0]
    max_len = max_label_length * multiplier
    
    if cur_len < max_len: 
        pad = tf.constant([[0, max_len-cur_len], [0,0], [0,0]])
        features = tf.pad(features, pad, 'constant')
    else:
        features = features[:max_len]
    return features



if __name__ == '__main__':
    ''' An example of how to use '''
    import os
    import time
    from transforms import *

    path = '/media/data1/datasets/DCASE2020/feat_label/'
    x, y = load_seldnet_data(os.path.join(path, 'foa_dev_norm'),
                             os.path.join(path, 'foa_dev_label'),
                             mode='val')

    sample_transforms = [
        lambda x, y: (mask(x, axis=-3, max_mask_size=24, n_mask=6), y),
        lambda x, y: (mask(x, axis=-2, max_mask_size=8), y),
    ]
    batch_transforms = [
        split_total_labels_to_sed_doa,
    ]
    dataset = seldnet_data_to_dataloader(
        x, y,
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms,
    )

    start = time.time()
    for i in range(10):
        for x, y in dataset:
            pass

        print(time.time() - start)
        start = time.time()
    