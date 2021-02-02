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

