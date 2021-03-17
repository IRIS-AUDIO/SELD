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
            return {'flop': 2, 'params': 5}, input_shape

        def block_b_complexity(args, input_shape):
            return {'flop': 7, 'params': 3}, input_shape

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

        gt = {'flop': 9, 'params': 8}
        self.assertEqual(
            gt, complexity(model_config, input_shape, mapping_dict))

    def test_dict_add(self):
        a = {'a': 3}
        b = {'a': 2, 'b': 4}

        gt = {'a': 5, 'b': 4}

        c = dict_add(a, b)
        self.assertEqual(c, gt)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

