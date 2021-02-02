import numpy as np
import tensorflow as tf


def data_loader(dataset, 
                sample_transforms=None, 
                batch_transforms=None,
                batch_size=32):
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if sample_transforms is not None:
        if not isinstance(sample_transforms, (list, tuple)):
            sample_transforms = [sample_transforms]

        for p in sample_transforms:
            dataset = dataset.map(p)

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if batch_transforms is not None:
        if not isinstance(batch_transforms, (list, tuple)):
            batch_transforms = [batch_transforms]

        for p in batch_transforms:
            dataset = dataset.map(p)

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

    features = sorted(glob(os.path.join(feat_path, '*.npy')))
    features = [np.load(f).astype('float32') for f in features 
                if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]
    # reshape [..., freq*chan] -> [..., freq, chan]
    features = [np.reshape(f, (*f.shape[:-1], n_freq_bins, -1))
                for f in features]
    
    labels = sorted(glob(os.path.join(label_path, '*.npy')))
    labels = [np.load(f) for f in labels
              if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    return features, labels


def seldnet_data_to_dataloader(features: [list, tuple], 
                               labels: [list, tuple], 
                               train=True, 
                               label_time_per_sample=60,
                               **kwargs):
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # shapes of features and labels 
    # features: [time_features, freq, chan]
    # labels:   [time_labels, 4*classes]
    # for each 5 input time slices, a single label time slices was designated
    
    # features' shape [time_f, freq, chan] -> [time_l, resolution, freq, chan]
    features = np.reshape(features, (labels.shape[0], -1, *features.shape[1:]))

    # windowing
    n_samples = features.shape[0] // label_time_per_sample
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(label_time_per_sample)
    dataset = dataset.map(lambda x,y: (tf.reshape(x, (-1, *x.shape[2:])), y))
    del features, labels

    dataset = data_loader(dataset, **kwargs)
    if train:
        dataset = dataset.shuffle(n_samples)

    return dataset


if __name__ == '__main__':
    ''' An example of how to use '''
    from transforms import *
    import matplotlib.pyplot as plt

    path = '/media/data1/datasets/DCASE2020/feat_label/'
    x, y = load_seldnet_data(path+'foa_dev_norm', path+'foa_dev_label', mode='val')

    sample_transforms = [
        lambda x, y: (mask(x, axis=-3, max_mask_size=24, n_mask=6), y),
        lambda x, y: (mask(x, axis=-2, max_mask_size=8), y),
    ]
    batch_transforms = [
        split_total_labels_to_sed_doa
    ]
    dataset = seldnet_data_to_dataloader(
        x, y,
        sample_transforms=sample_transforms,
        batch_transforms=batch_transforms,
    )

    def norm(xs):
        return (xs - tf.reduce_min(xs)) / (tf.reduce_max(xs) - tf.reduce_min(xs))

    for x, y in dataset:
        fig, axs = plt.subplots(2)
        axs[0].imshow(norm(x[0])[..., 0]) # tf.reshape(norm(x), (x.shape[0], -1)))
        axs[1].imshow(y[1][0])
        plt.show()

