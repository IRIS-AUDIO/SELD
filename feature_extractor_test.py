import os
import tensorflow as tf
import torch
from feature_extractor import *


class FeatureExtractorTest(tf.test.TestCase):
    def setUp(self):
        self.cartesians = [ # x, y, z
            [ 0,  0,  1],
            [ 0, -1,  0],
            [ 1,  0,  0],
            [-2,  2,  0],
            [ 0,  0,  0],
        ]
        self.polars = [ # azimuth, elevation, r
            [   0,  90,   1],
            [ -90,   0,   1],
            [   0,   0,   1],
            [ 135,   0, np.sqrt(8)],
            [   0,   0,   0],
        ]

    def test_extract_features(self):
        wav = torch.zeros((4, 32000))
        r = 16000

        foa = extract_features(wav, r, mode='foa')
        self.assertEqual(foa.ndim, 3)
        self.assertEqual(foa.shape[-1], 7) # channel

        foa = extract_features(wav, r, mode='mic')
        self.assertEqual(foa.ndim, 3)
        self.assertEqual(foa.shape[-1], 10) # channel

    def test_cartesian_to_polar(self):
        self.assertAllClose(
            cartesian_to_polar(self.cartesians),
            self.polars,
        )

    def test_polar_to_cartesian(self):
        self.assertAllClose(
            polar_to_cartesian(self.polars),
            self.cartesians,
        )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

