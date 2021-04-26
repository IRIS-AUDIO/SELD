import numpy as np
import tensorflow as tf
from feature_extractor import *
import random as rnd
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
    tdm_x = [torchaudio.load(f)[0] for f in sorted(glob(TDM_PATH + '/single_sound/*'))]
    tdm_y = [np.load(f) for f in sorted(glob(TDM_PATH + '/single_label/*'))]
    return tdm_x, tdm_y

def TDM_aug(x, y, tdm_x, tdm_y):
    for x_, y_ in zip(x,y):
        single_source = y_[:,:14]
        check_single = np.sum(single_source, axis=1)
        single_index = np.where(check_single == 1)
        
        check_same_label = 0
        check_sequence = 0
        index_list = []
        
        frame_len = 0 # length of start frame
        start_location = 0 # start location of specific class
        new_location = 0 # check weather start frame changed
    
        sequence = [i for i in range(len(tdm_x))]
        for single in single_index[0]:        
            if new_location == 0:
                check_sequence = single - 1
                check_same_label = np.argwhere(single_source[single] == 1)[0][0]
                start_location = single
                frame_len = 1
                new_location = 1
                
            if (single - 1) == check_sequence and \
                check_same_label == np.argwhere(single_source[single] == 1)[0][0]:
                check_sequence = single
                frame_len += 1
                
            else:
                if frame_len >= 10 : 
                    index_list.append([start_location, frame_len, check_same_label])
                new_location = 0
        
        for index in index_list: 
            pick = rnd.choice(tdm_y[sequence])
            
            while(np.argwhere(pick[0,:14] == 1)[0][0] == index[2]):
                pick = rnd.choice(tdm_y[sequence])
                
            if rnd.random() > 0.5 : 
                length_diff = index[1] - len(tdm_y)
                noise_ = rnd.random()*0.4 + 0.3
                if length_diff > 0 : 
                    offset = int(rnd.random() * length_diff)
                    mix_x = tdm_x
                    mix_y = tdm_y
    
                    x_[:, offset + index[0]:offset + index[0] + len(mix_y)] = \
                    mix_x * noise_ + \
                    x_[:, offset + index[0]:offset + index[0] + len(mix_y)] * (1 - noise_)
                    
                    y_[offset + index[0]:offset + index[0] + index[1],:] =\
                    y_[offset + index[0]:offset + index[0] + index[1],:] + mix_y
    
                else:
                    offset = int(rnd.random() * (-length_diff))
                    mix_x = tdm_x[:, offset:offset + index[1]]
                    mix_y = tdm_y[offset:offset + index[1], :]
    
                    x_[:, index[0]:index[0] + index_1] = \
                    mix_x*noise_ + \
                    x_[:, index[0]:index[0] + len(mix_y)]*(1 - noise_)

                    y_[index[0]:index[0]+index[1], :] = \
                    y_[index[0]:index[0]+index[1], :] + mix_y
    return x, y    

        
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
    