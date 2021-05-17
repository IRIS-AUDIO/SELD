import numpy as np
import tensorflow as tf
from feature_extractor import *
import random as rnd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
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


def get_preprocessed_wave(feat_path, label_path, mode='train'):
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
            labels = np.pad(labels, ((0, max_len-cur_len), (0,0)), 'constant')
        else:
            labels = labels[:max_len]
        return labels
    
    x = list(map(lambda x: torchaudio.load(x)[0], f_paths))
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
    device = get_device()

    def load_data(cls):
        return torch.from_numpy(joblib.load(os.path.join(tdm_path, f'tdm_noise_{cls}.joblib'))).to(device)

    def load_label(cls):
        return torch.from_numpy(joblib.load(os.path.join(tdm_path, f'tdm_label_{cls}.joblib'))).to(device)
    
    with ThreadPoolExecutor() as pool:
        tdm_x = list(pool.map(load_data, range(class_num)))
        tdm_y = list(pool.map(load_label, range(class_num)))
    return tdm_x, tdm_y


def TDM_aug(x: list, y: list, tdm_x, tdm_y, sr=24000, label_resolution=0.1, max_overlap_num=5, max_overlap_per_frame=2, min_overlap_time=1, max_overlap_time=5):
    '''
        x: list(torch.Tensor): shape(sample number, channel(4), frame(1440000))
        y: list(np.ndarray): shape(sample number, time(600), class+cartesian(14+42))
        tdm_x: list(np.ndarray): shape(class_num, channel(4), frame)
        tdm_y: list(np.ndarray): shape(class_num, time, class+cartesian(14+42))
    '''
    class_num = y[0].shape[-1] // 4
    min_overlap_time *= int(1 / label_resolution) #  
    max_overlap_time *= int(1 / label_resolution) # 
    sr = int(sr * label_resolution)
    device = get_device()
    if tdm_x[0].device != device:
        tdm_x = list(map(lambda x: x.to(device), tdm_x))
        tdm_y = list(map(lambda x: x.to(device), tdm_y))

    weight = 1 / torch.tensor([i.shape[0] for i in tdm_y])
    weight /= weight.sum()
    weight = weight.cumsum(-1)
    def add_noise(i):
        selected_cls = weight.multinomial(max_overlap_num, replacement=True) # (max_overlap_num,)

        def _add_noise(cls):
            xs, ys = x[i].to(device), torch.from_numpy(y[i]).to(device)

            td_x = tdm_x[cls].type(xs.dtype)
            td_y = tdm_y[cls].type(ys.dtype)
            sample_time = torch.randint(min_overlap_time, max_overlap_time, (1,), device=xs.device) # to milli second
            offset = torch.randint(ys.shape[0] - sample_time.item(), (1,), device=xs.device) # offset as label

            nondup_class = 1 - ys[..., cls] # 프레임 중 class가 겹치지 않는 부분 찾기
            
            valid_index = torch.where(torch.logical_and(ys[...,:class_num].sum(-1) < max_overlap_per_frame, nondup_class))[0].to(xs.device) # 1프레임당 최대 클래스 개수보다 작으면서 겹치지 않는 노이즈를 넣을 수 있는 공간 찾기

            frame_idx = torch.arange(sample_time.item(), device=xs.device) # sample_time 크기만한 frame idx 생성
            y_idx = frame_idx + offset # 합칠 프레임들 전체
            con = (torch.unsqueeze(y_idx, -1) == valid_index).sum(-1) # valid_index 중 y_idx에 있는 idx만 찾기, masking 방식의 결과
            if con.sum() == 0: # 만약 넣을 수 없다면 이번에는 노이즈 안 넣음
                return
            
            idx = torch.where(con > 0)[0].to(xs.device)
            y_idx = y_idx[idx].cpu() # 해당되지 않는 부분 삭제, 자리 확정
            td_offset = torch.randint(td_y.shape[0] - sample_time.item(), (1,), device=xs.device) # 뽑을 노이즈에서의 랜덤 offset
            td_y_idx = idx + td_offset # 뽑을 노이즈 index
            
            x_idx = torch.cat([torch.arange((i * sr).item(), (i * sr + sr).item(), dtype=torch.long, device=xs.device) for i in y_idx]).cpu()
            td_x_idx = torch.cat([torch.arange((i * sr).item(), (i * sr + sr).item(), dtype=torch.long, device=xs.device) for i in td_y_idx])

            x[i][..., x_idx] = (xs[..., x_idx] + td_x[..., td_x_idx]).cpu()
            y[i][y_idx] = (ys[y_idx] + td_y[td_y_idx]).cpu() # 레이블 부분 완료
        
        list(map(_add_noise, selected_cls))
    # with ThreadPoolExecutor(5) as pool:
    #     list(pool.map(add_noise, tqdm(range(len(x)))))
    list(map(add_noise, tqdm(range(len(x)))))
    if tdm_x[0].device != 'cpu':
        tdm_x = list(map(lambda x: x.cpu(), tdm_x))
        tdm_y = list(map(lambda x: x.cpu(), tdm_y))
    return x, y


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
    