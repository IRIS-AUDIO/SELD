import os
import tensorflow as tf
from collections import OrderedDict
from config_sampler import *


class ConfigSamplerTest(tf.test.TestCase):
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

    def test_conv_temporal_sampler(self):
        search_space_2d = {
            'conv': {'filters': [16, 32, 64], 'use_bias': [True, False]},
            'block': {'depth': [2, 4, 8]}
        }
        search_space_1d = {
            'linear': {'activation': ['relu', None], 'units': [8, 16]},
            'gru': {'units': [39]}
        }
        default_config = {'n_classes': 10}
        model_config = conv_temporal_sampler(search_space_2d,
                                             search_space_1d,
                                             n_blocks=4,
                                             input_shape=[7, 80, 1],
                                             default_config=default_config,
                                             constraint=None)

    def test_vad_architecture_sampler(self):
        search_space_2d = {
            'conv': {'filters': [16, 32, 64], 'use_bias': [True, False]},
            'block': {'depth': [2, 4, 8]}
        }
        search_space_1d = {
            'linear': {'activation': ['relu', None], 'units': [8, 16]},
            'gru': {'units': [39]}
        }
        default_config = {'last_units': 1}
        model_config = vad_architecture_sampler(search_space_2d,
                                                search_space_1d,
                                                n_blocks=4,
                                                input_shape=[7, 80, 1],
                                                default_config=default_config,
                                                constraint=None)

    def test_search_space_sanity_check(self):
        clean = {'typeA': {'a': [1, 3, 5], 'b': (2,)}}
        dirty = {'typeB': {'b': [], 'c': 3}}

        search_space_sanity_check(clean)
        with self.assertRaises(ValueError):
            search_space_sanity_check(dirty)

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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

