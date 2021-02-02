import numpy as np
import tensorflow as tf



def data_loader(data, 
                sample_transforms=None, 
                batch_transforms=None,
                batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(data)

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
    features = [np.load(f) for f in features 
                if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]
    # reshape [..., freq*chan] -> [..., freq, chan]
    features = [np.reshape(f, (*f.shape[:-1], n_freq_bins, -1))
                for f in features]
    
    labels = sorted(glob(os.path.join(label_path, '*.npy')))
    labels = [np.load(f) for f in labels
              if int(f[f.rfind(os.path.sep)+5]) in splits[mode]]

    return features, labels

