import os
import tensorflow as tf
from complexity import *


class ComplexityTest(tf.test.TestCase):
    def setUp(self):
        self.prev_cx = {'flops': 456, 'params': 123}

    def test_simple_conv_block_complexity(self):
        model_config = {
            'filters': [32, 32],
            'pool_size': [2, 2],
        }
        input_shape = [32, 32, 16]
        self.assertEqual(
            simple_conv_block_complexity(model_config, input_shape),
            ({'flops': 7077952, 'params': 13888}, [8, 8, 32]))

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
        target_cx = {'flops': 442384, 'params': 448}
        target_shape = [32, 32, 16]

        self.assertEqual(
            conv2d_complexity(input_shape=[32, 32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1),
            (target_cx, target_shape))
        self.assertEqual(
            conv2d_complexity(input_shape=[32, 32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1,
                              prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))

    def test_norm_complexity(self):
        target_cx = {'params': 6}
        target_shape = [32, 32, 3]

        self.assertEqual(
            norm_complexity(input_shape=[32, 32, 3],
                            center=True,
                            scale=True),
            (target_cx, target_shape))
        self.assertEqual(
            norm_complexity(input_shape=[32, 32, 3],
                            center=True,
                            scale=True,
                            prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))

    def test_pool2d_complexity(self):
        target_cx = {}
        target_shape = [16, 32, 3]

        self.assertEqual(
            pool2d_complexity(input_shape=[32, 32, 3],
                              pool_size=3,
                              strides=(2, 1)),
            (target_cx, target_shape))
        self.assertEqual(
            pool2d_complexity(input_shape=[32, 32, 3],
                              pool_size=3,
                              strides=(2, 1),
                              prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))

    def test_linear_complexity(self):
        target_cx = {'flops': 1050624, 'params': 525312}
        target_shape = [2, 1024]

        self.assertEqual(
            linear_complexity(input_shape=[2, 512],
                              units=1024,
                              use_bias=True),
            (target_cx, target_shape))
        self.assertEqual(
            linear_complexity(input_shape=[2, 512],
                              units=1024,
                              use_bias=True,
                              prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))

    def test_GRU_complexity(self):
        target_cx = {'flops': 978000, 'params': 9360}
        target_shape = [32, 100, 30]
        self.assertEqual(
            gru_complexity(input_shape=[32, 100, 20],
                           units=30,
                           use_bias=True,
                           bi=True),
            (target_cx, target_shape))
        self.assertEqual(
            gru_complexity(input_shape=[32, 100, 20],
                           units=30,
                           use_bias=True,
                           bi=True,
                           prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))
        
    def test_multi_head_attention(self):
        target_cx = {'flops': 109785600, 'params': 790656}
        target_shape = [100, 128]
        self.assertEqual(
            multi_head_attention_complexity(input_shape=[100, 128],
                                 num_heads=4,
                                 key_dim=256,
                                 value_dim=512,
                                 use_bias=True),
            (target_cx, target_shape))
        self.assertEqual(
            multi_head_attention_complexity(input_shape=[100, 128],
                                 num_heads=4,
                                 key_dim=256,
                                 value_dim=512,
                                 use_bias=True,
                                 prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))
    
    def test_safe_tuple(self):
        self.assertEqual((1, 1), safe_tuple(1, 2))
        self.assertEqual((1, 3), safe_tuple((1, 3), 2))
        with self.assertRaises(ValueError):
            safe_tuple((1, 2, 3), 2)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

