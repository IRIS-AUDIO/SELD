import os
import tensorflow as tf
from config_sampling import *


class ConfigSamplingTest(tf.test.TestCase):
    def test_config_sampling(self):
        search_space = {
            'FIRST': ['A', 'B'],
            'FIRST_ARGS': {
                'A': {'a': [1, 2, 3]},
                'B': {'1': [0, 1, 2], '2': [1, 3, 5]},
            }
        }

        sample = config_sampling(search_space)
        self.assertTrue(sample['FIRST'] in search_space['FIRST'])
        self.assertTrue(search_space['FIRST_ARGS'][sample['FIRST']].keys() 
                        == sample['FIRST_ARGS'].keys())
        for key in sample['FIRST_ARGS'].keys():
            self.assertTrue(sample['FIRST_ARGS'][key] in 
                            search_space['FIRST_ARGS'][sample['FIRST']][key])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

