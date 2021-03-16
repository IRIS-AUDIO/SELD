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
        model_config = OrderedDict({
            'FIRST': 'A', 
            'FIRST_ARGS': { 'a': 1, },
            'SECOND': 'B',
            'SECOND_ARGS': { 'B': 35, 'b': 50 }
        })
        input_shape = (32, 32, 3) # without batch dim

        # Mapping Dict
        # key: block type
        # value: func(block_args, input_shape) -> FLOPs, output_shape
        mapping_dict = {
            'A': lambda c, s: (input_shape[-1]*c['a'], input_shape),
            'B': lambda c, s: (input_shape[-2]*c['B'] + c['b'], input_shape),
        }

        self.assertTrue(complexity(model_config, input_shape, mapping_dict) > 0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

