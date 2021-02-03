import os
import tensorflow as tf
from data_loader import *


class DataLoaderTest(tf.test.TestCase):
    def test_data_loader(self):
        # data init
        data_size = 16
        batch_size = 8
        xs = tf.range(data_size)
        ys = tf.range(data_size, 0, -1)

        sample_transforms = [lambda x, y: (x, y)]
        batch_transforms = [lambda x, y: (x, y)]

        dataset = data_loader(
            (xs, ys), 
            sample_transforms,
            batch_transforms,
            batch_size=batch_size)

        # test
        for x, y in dataset:
            self.assertEqual(x.shape, [batch_size])
            self.assertEqual(y.shape, [batch_size])

    def test_load_seldnet_data(self):
        #TODO: change real files to temp files
        default_path = '/media/data1/datasets/DCASE2020/feat_label/'
        feat_path = os.path.join(default_path,
                                 'foa_dev_norm')
        label_path = os.path.join(default_path,
                                  'foa_dev_label')

        # parameters for SELDnet data
        freq = 64
        chan = 7
        n_classes = 14

        x, y = load_seldnet_data(feat_path, label_path, 
                                 mode='val')

        self.assertEqual(len(x), len(y))
        self.assertEqual(x[0].shape[-2:], (freq, chan))
        self.assertEqual(y[0].shape[-1], n_classes*4)

    def test_seldnet_data_to_dataloader(self):
        # shape of x
        n_samples, time_x, freq, chan = 8, 80, 40, 7
        # shape of y
        _, time_y, n_classes = n_samples, 16, 11

        x = [tf.zeros((time_x, freq, chan)) for _ in range(n_samples)]
        y = [tf.zeros((time_y, n_classes*4)) for _ in range(n_samples)]

        # shapes of processed x and y
        # shape of x: [batch, label_window_size*5, freq, chan]
        # shape of y: [batch, label_window_size, n_classes*4]
        label_window_size = 8

        dataset = seldnet_data_to_dataloader(
            x, y, label_window_size=label_window_size, batch_size=n_samples)

        for x, y in dataset:
            self.assertEqual(x.shape, 
                             [n_samples, label_window_size*5, freq, chan])
            self.assertEqual(y.shape,
                             [n_samples, label_window_size, n_classes*4])
        

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()
