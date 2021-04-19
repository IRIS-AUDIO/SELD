import os
import tensorflow as tf
import tensorflow.keras.backend as K
from modules import *
from complexity import *


class ComplexityTest(tf.test.TestCase):
    def setUp(self):
        self.prev_cx = {'flops': 456, 'params': 123}

    def test_simple_conv_block_complexity(self):
        model_config = {
            'filters': [32, 32],
            'pool_size': [2, 2],
        }
        self.complexity_test(simple_conv_block_complexity,
                             simple_conv_block,
                             model_config,
                             [32, 32, 16])

    def test_another_conv_block_complexity(self):
        model_config = {
            'filters': 32,
            'depth': 3,
            'pool_size': [2, 1],
        }
        self.complexity_test(another_conv_block_complexity,
                             another_conv_block,
                             model_config,
                             [32, 32, 16])

    def test_res_basic_stage_complexity(self):
        model_config = {
            'depth': 4,
            'strides': 2,
            'filters': 24,
        }
        self.complexity_test(res_basic_stage_complexity,
                             res_basic_stage,
                             model_config,
                             [32, 32, 16])

    def test_res_basic_block_complexity(self):
        model_config = {
            'strides': 2,
            'filters': 24,
        }
        self.complexity_test(res_basic_block_complexity,
                             res_basic_block,
                             model_config,
                             [32, 32, 3])

    def test_res_bottleneck_stage_complexity(self):
        model_config = {
            'depth': 4,
            'strides': 2,
            'filters': 24,
            'groups': 2,
            'bottleneck_ratio': 2
        }
        self.complexity_test(res_bottleneck_stage_complexity,
                             res_bottleneck_stage,
                             model_config,
                             [32, 32, 16])

    def test_res_bottleneck_block_complexity(self):
        model_config = {
            'filters': 32,
            'strides': 2,
            'groups': 2,
            'bottleneck_ratio': 2
        }
        self.complexity_test(res_bottleneck_block_complexity,
                             res_bottleneck_block,
                             model_config,
                             [32, 32, 3])

    def test_dense_net_block_complexity(self):
        model_config = {
            'growth_rate': 8,
            'depth': 3,
            'strides': 2,
            'bottleneck_ratio': 2,
            'reduction_ratio': 0.5,
        }
        self.complexity_test(dense_net_block_complexity,
                             dense_net_block,
                             model_config,
                             [32, 32, 3])

    def test_sepformer_block_complexity(self):
        model_config = {
            'n_head': 8,
            'ff_multiplier': 4,
            'kernel_size': 3,
        }
        self.complexity_test(sepformer_block_complexity,
                             sepformer_block,
                             model_config,
                             [32, 32, 3])

    def test_xception_block_complexity(self):
        model_config = {
            'filters': 32,
            'block_num': 1,
        }
        self.complexity_test(xception_block_complexity,
                             xception_block,
                             model_config,
                             [32, 32, 3])

    def test_bidirectional_GRU_block_complexity(self):
        model_config = {
            'units': [128, 128],
        }
        self.complexity_test(bidirectional_GRU_block_complexity,
                             bidirectional_GRU_block,
                             model_config,
                             [32, 32, 3])

    def test_transformer_encoder_block_complexity(self):
        model_config = {
            'n_head': 4,
            'ff_multiplier': 2,
            'kernel_size': 3,
        }
        self.complexity_test(transformer_encoder_block_complexity,
                             transformer_encoder_block,
                             model_config,
                             [32, 48])

    def test_simple_dense_block_complexity(self):
        model_config = {
            'units': [32, 32]
        }
        self.complexity_test(simple_dense_block_complexity,
                             simple_dense_block,
                             model_config,
                             [32, 48])

    def test_identity_block_complexity(self):
        model_config = {}
        self.complexity_test(identity_block_complexity,
                             identity_block,
                             model_config,
                             [32, 32, 48])


    def test_conformer_block_complexity(self):
        model_config = {
            'n_head': 8,
            'multiplier': 4,
            'key_dim': 36,
            'kernel_size': 32,
        }
        self.complexity_test(conformer_block_complexity,
                             conformer_encoder_block,
                             model_config,
                             [100, 40])

    def test_conv1d_complexity(self):
        target_cx = {'flops': 4608, 'params': 160}
        target_shape = [32, 16]

        self.assertEqual(
            conv1d_complexity(input_shape=[32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1),
            (target_cx, target_shape))
        self.assertEqual(
            conv1d_complexity(input_shape=[32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1,
                              prev_cx=self.prev_cx),
            (dict_add(target_cx, self.prev_cx), target_shape))


    def test_conv2d_complexity(self):
        target_cx = {'flops': 442368, 'params': 448}
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

    def test_separable_conv2d_complexity(self):
        args = {
            'filters': 128,
            'kernel_size': 3,
            'strides': (1, 2),
            'padding': 'valid',
            'use_bias': True,
            'depth_multiplier': 2,
        }
        input_shape = [6, 32, 64]

        inputs = tf.keras.layers.Input(input_shape)
        outputs = tf.keras.layers.SeparableConv2D(**args)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        cx, shape = separable_conv2d_complexity(input_shape, **args)
        self.assertEqual(
            cx['params'], 
            sum([K.count_params(p) for p in model.trainable_weights]))
        self.assertEqual(tuple(shape), model.output_shape[1:])

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

    def complexity_test(self, 
                        complexity_fn,
                        block_fn,
                        model_config: dict,
                        exp_input_shape: list):
        '''
        complexity_fn: a func that calculates the complexity
                       of given block(stage)
        block_fn: a func that will generate the block
        model_config: model_config for block_fn
        exp_input_shape: expected input shape
        exp_output_shape: expected output_shape

        "batch size" must not be included in both arguments
        ex) [time, chan]
        '''
        inputs = tf.keras.layers.Input(exp_input_shape)
        outputs = block_fn(model_config)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        cx, output_shape = complexity_fn(model_config, exp_input_shape)

        # TODO: count ops
        self.assertEquals(cx['params'], 
                          sum([K.count_params(p) 
                               for p in model.trainable_weights]))
        self.assertEquals(tuple(output_shape), model.output_shape[1:])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

