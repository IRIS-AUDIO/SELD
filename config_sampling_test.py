import os
import tensorflow as tf
from collections import OrderedDict
from config_sampling import *


class ConfigSamplingTest(tf.test.TestCase):
    def test_config_sampling(self):
        search_space = OrderedDict({
            'FIRST': ['A', 'B'],
            'FIRST_ARGS': {
                'A': {'a': [1, 2, 3]},
                'B': {'1': [0, 1, 2], '2': [1, 3, 5]},
            },
        })

        sample = config_sampling(search_space)
        self.assertTrue(sample['FIRST'] in search_space['FIRST'])
        self.assertTrue(search_space['FIRST_ARGS'][sample['FIRST']].keys() 
                        == sample['FIRST_ARGS'].keys())
        for key in sample['FIRST_ARGS'].keys():
            self.assertTrue(sample['FIRST_ARGS'][key] in 
                            search_space['FIRST_ARGS'][sample['FIRST']][key])

    def test_complexity(self):
        # complexity calculating functions (for the test)
        def block_a_complexity(args, input_shape):
            return {'flops': 2, 'params': 5}, input_shape

        def block_b_complexity(args, input_shape):
            return {'flops': 7, 'params': 3}, input_shape

        # inputs for complexity (func)
        model_config = OrderedDict({
            'FIRST': 'A', 
            'FIRST_ARGS': { 'a': 1, },
            'SECOND': 'B',
            'SECOND_ARGS': { 'B': 35, 'b': 50 }
        })
        input_shape = (32, 32, 3) # without batch dim
        mapping_dict = {
            'A': block_a_complexity,
            'B': block_b_complexity,
        }

        gt = {'flops': 9, 'params': 8}
        self.assertEqual(
            gt, complexity(model_config, input_shape, mapping_dict))

    def test_dict_add(self):
        a = {'a': 3}
        b = {'a': 2, 'b': 4}

        gt = {'a': 5, 'b': 4}

        c = dict_add(a, b)
        self.assertEqual(c, gt)

    def test_safe_tuple(self):
        self.assertEqual((1, 1), safe_tuple(1, 2))
        self.assertEqual((1, 3), safe_tuple((1, 3), 2))
        with self.assertRaises(ValueError):
            safe_tuple((1, 2, 3), 2)

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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

