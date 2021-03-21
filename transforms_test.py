import os
import numpy as np
import tensorflow as tf
from transforms import *


class TransformsTest(tf.test.TestCase):
    def test_mask(self):
        tf.random.set_seed(100)
        org = np.array([[ 0,  1,  2,  3,  4],
                        [ 5,  6,  7,  8,  9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24]])
        target = np.array([[ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [15, 16, 17, 18, 19],
                           [20, 21, 22, 23, 24]])
        self.assertAllEqual(target, 
                            mask(org, axis=0, max_mask_size=None, n_mask=1))

        tf.random.set_seed(2020)
        target = np.array([[ 0,  1,  0,  3,  4],
                           [ 0,  6,  0,  8,  9],
                           [ 0, 11,  0, 13, 14],
                           [ 0, 16,  0, 18, 19],
                           [ 0, 21,  0, 23, 24]])
        self.assertAllEqual(target, 
                            mask(org, axis=1, max_mask_size=3, n_mask=2))

    def test_intensity_vec_aug(self):
        tf.random.set_seed(2022)
        x_size = (1, 10, 32, 7) # [batch, time, freq, 7]
        y_size = (1, 2, 12) # [batch, time, n_classes*4]
        x = tf.random.uniform(x_size)
        y = tf.random.uniform(y_size) 

        # test output shapes
        new_x, new_y = foa_intensity_vec_aug(x, y)
        self.assertAllEqual(x.shape, x_size)
        self.assertAllEqual(y.shape, y_size)

        # test whether x and y are fliped equally
        x_flip = x[..., -3:] != new_x[..., -3:]
        x_flip = tf.reduce_mean(tf.cast(x_flip, 'float32'), axis=(1, 2))

        y_flip = tf.reshape(y, y_size[:-1]+(4, -1))[..., -3:, :] \
                != tf.reshape(new_y, y_size[:-1]+(4, -1))[..., -3:, :]
        y_flip = tf.reduce_mean(tf.cast(y_flip, 'float32'), axis=(1, 3))

        self.assertAllEqual(x_flip, y_flip)

    def test_split_total_labels_to_sed_doa(self):
        batch, time, n_classes = 32, 10, 14
        y = tf.zeros((batch, time, n_classes * 4))

        _, (sed, doa) = split_total_labels_to_sed_doa(None, y)
        self.assertAllEqual(sed.shape,
                            [batch, time, n_classes])
        self.assertAllEqual(doa.shape,
                            [batch, time, n_classes*3])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()
