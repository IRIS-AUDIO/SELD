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
    tdm_x = [torchaudio.load(f)[0] for f in sorted(glob(TDM_PATH + '/single_sound/*'))]
    tdm_y = [np.load(f) for f in sorted(glob(TDM_PATH + '/single_label/*'))]
    sr = torchaudio.load(sorted(glob(TDM_PATH + '/single_sound/*'))[0])[1]
    return tdm_x, tdm_y, sr


def TDM_aug(x, y, tdm_x, tdm_y, sr):
    from tqdm import tqdm
    for x_, y_ in tqdm(zip(x,y)):
        single_source = y_[:,:14]
        # check If source is single
        check_single = np.sum(single_source, axis=1)
        single_index = np.where(check_single == 1)
        
        check_same_label = 0 # check It is same label
        check_sequence = 0 # check if this segment is sequnetial
        index_list = [] # will store start, length, label class
        
        frame_len = 0 # length of start frame
        start_location = 0 # start location of specific class
        new_location = 0 # check weather start frame changed
    
        sequence = [i for i in range(len(tdm_x))]
        for single in single_index[0]:        
            if new_location == 0: # When first frame of specific source
                # Set condition for first frame of sequence
                check_sequence = single - 1
                check_same_label = np.argwhere(single_source[single] == 1)[0][0]
                start_location = single
                frame_len = 1
                new_location = 1
                
            # If cascade frame has same class, measure length of sequence
            if (single - 1) == check_sequence and \
                check_same_label == np.argwhere(single_source[single] == 1)[0][0]:
                check_sequence = single
                frame_len += 1
                new_location = 1
            
            # If sequence is ended, check length of sequence
            else:
                if frame_len >= 10 : 
                    index_list.append([start_location, frame_len, check_same_label])
                new_location = 0

        for index in index_list: 
            
            # Pick Some segment from single source
            rnd_num = rnd.choice(sequence)
            pick_y = tdm_y[rnd_num]
            
            # check weather class is same 
            while(np.argwhere(pick_y[0,:14] == 1)[0][0] == index[2]):
                rnd_num = rnd.choice(sequence)
                pick_y = tdm_y[rnd_num]
            pick_x = tdm_x[rnd_num]


            if rnd.random() > 0.5 :  # set probability
                length_diff = index[1] - len(pick_y)
                noise_ = rnd.random()*0.4 + 0.3 # mix with weight
                if length_diff > 0 : # case when mixing sound is shorter
                    offset = int(rnd.random() * length_diff) #set random offset
                    mix_x = pick_x
                    mix_y = pick_y
    
                    x_[:, int(0.1*sr*(offset + index[0])):int(0.1*sr*(offset + index[0] + len(mix_y)))] = \
                    mix_x * noise_ + \
                    x_[:, int(0.1*sr*(offset + index[0])):int(0.1*sr*(offset + index[0] + len(mix_y)))] * (1 - noise_)
                    
                    y_[offset + index[0]:offset + index[0] + len(mix_y),:] =\
                    y_[offset + index[0]:offset + index[0] + len(mix_y),:] + mix_y
    
                else:  # case when mixing sound is longer
                    offset = int(rnd.random() * (-length_diff))  # set random offset
                    mix_x = pick_x[:, int(0.1*sr*offset):int(0.1*sr*(offset + index[1]))]
                    mix_y = pick_y[offset:offset + index[1], :]
    
                    x_[:, int(0.1*sr*index[0]):int(0.1*sr*(index[0] + index[1]))] = \
                    mix_x*noise_ + \
                    x_[:, int(0.1*sr*index[0]):int(0.1*sr*(index[0] + index[1]))]*(1 - noise_)

                    y_[index[0]:index[0]+index[1], :] = \
                    y_[index[0]:index[0]+index[1], :] + mix_y
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
    