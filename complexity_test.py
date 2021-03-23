import os
import tensorflow as tf
from complexity import *


class ComplexityTest(tf.test.TestCase):
    def test_res_bottleneck_block_complexity(self):
        model_config = {
            'filters': 32,
            'strides': 2,
            'groups': 2,
            'bottleneck_ratio': 2
        }
        input_shape = [32, 32, 16]
        self.assertEqual(
            res_bottleneck_block_complexity(model_config, input_shape),
            ({'flops': 6422720, 'params': 22592}, [16, 16, 32]))

    def test_conv2d_complexity(self):
        self.assertEqual(
            conv2d_complexity(input_shape=[32, 32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1),
            ([32, 32, 16], {'flops': 442384, 'params': 448}))

    def test_norm_complexity(self):
        self.assertEqual(
            norm_complexity(input_shape=[32, 32, 3],
                              center=True,
                              scale=True),
            ([32, 32, 3], {'params': 6}))

    def test_pool2d_complexity(self):
        self.assertEqual(
            pool2d_complexity(input_shape=[32, 32, 3],
                               pool_size=3,
                               strides=(2, 1)),
            ([16, 32, 3], {}))

    def test_linear_complexity(self):
        self.assertEqual(
            linear_complexity(input_shape=[1, 512],
                              units=1024,
                              use_bias=True),
            ([1, 1024], {'flops': 1048576, 'params': 525312}))

    def test_safe_tuple(self):
        self.assertEqual((1, 1), safe_tuple(1, 2))
        self.assertEqual((1, 3), safe_tuple((1, 3), 2))
        with self.assertRaises(ValueError):
            safe_tuple((1, 2, 3), 2)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

