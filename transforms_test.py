import os
import torch
import torchaudio
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
