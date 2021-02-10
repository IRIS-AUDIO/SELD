import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def data_loader(dataset, 
                preprocessing=None,
                sample_transforms=None, 
                batch_transforms=None,
                deterministic=False,
                inf_loop=False,
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
    if inf_loop:
        dataset = dataset.repeat()
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
    labels = [np.load(f) for f in labels
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


def seldnet_data_to_dataloader(features: [list, tuple], 
                               labels: [list, tuple], 
                               train=True, 
                               label_window_size=60,
                               drop_remainder=True,
                               shuffle_size=None,
                               batch_size=32,
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

    dataset = data_loader(dataset, batch_size=batch_size, **kwargs)
    if train:
        if shuffle_size is None:
            shuffle_size = n_samples // batch_size
        dataset = dataset.shuffle(shuffle_size)

    return dataset.prefetch(AUTOTUNE)


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

