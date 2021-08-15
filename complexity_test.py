import os
import tensorflow as tf
import tensorflow.keras.backend as K
from modules import *
from complexity import *


class ComplexityTest(tf.test.TestCase):
    def setUp(self):
        self.prev_cx = {'flops': 456, 'params': 123}

    def test_mother_block_complexity(self):
        model_config = {
            'filters0': 32,
            'filters1': 32,
            'filters2': 0,
            'kernel_size0': 3,
            'kernel_size1': 5,
            'kernel_size2': 0,
            'connect0': [1],
            'connect1': [1, 1],
            'connect2': [0, 0, 1],
            'strides': (1, 1),
        }
        self.complexity_test(mother_block_complexity,
                             mother_block,
                             model_config,
                             [32, 32, 16])

        # with SE
        model_config = {
            'filters0': 32,
            'filters1': 32,
            'filters2': 0,
            'kernel_size0': 3,
            'kernel_size1': 5,
            'kernel_size2': 0,
            'connect0': [1],
            'connect1': [1, 1],
            'connect2': [0, 0, 1],
            'strides': (1, 1),
            'squeeze_ratio': 0.5,
        }
        self.complexity_test(mother_block_complexity,
                             mother_block,
                             model_config,
                             [32, 32, 16])

    def test_bidirectional_GRU_block_complexity(self):
        model_config = {
            'units': [128, 128],
        }
        self.complexity_test(bidirectional_GRU_block_complexity,
                             bidirectional_GRU_block,
                             model_config,
                             [32, 32, 3])

    def test_RNN_block_complexity(self):
        model_config = {
            'units': 128,
            'bidirectional': True,
            'merge_mode': 'concat',
            'rnn_type': 'GRU',
        }
        self.complexity_test(RNN_block_complexity,
                             RNN_block,
                             model_config,
                             [12, 64])

        model_config = {
            'units': 128,
            'bidirectional': True,
            'merge_mode': 'ave',
            'rnn_type': 'LSTM',
        }
        self.complexity_test(RNN_block_complexity,
                             RNN_block,
                             model_config,
                             [12, 64])

    def test_transformer_encoder_block_complexity(self):
        model_config = {
            'n_head': 4,
            'key_dim': 16,
            'ff_multiplier': 2,
            'kernel_size': 3,
        }
        self.complexity_test(transformer_encoder_block_complexity,
                             transformer_encoder_block,
                             model_config,
                             [32, 48])

    def test_simple_dense_block_complexity(self):
        # ndim: 3
        model_config = {
            'units': [32, 32],
            'kernel_size': 2,
        }
        self.complexity_test(simple_dense_block_complexity,
                             simple_dense_block,
                             model_config,
                             [32, 48])

        # ndim: 2
        model_config = {
            'units': [32, 32],
        }
        self.complexity_test(simple_dense_block_complexity,
                             simple_dense_block,
                             model_config,
                             [32])

    def test_identity_block_complexity(self):
        model_config = {}
        self.complexity_test(identity_block_complexity,
                             identity_block,
                             model_config,
                             [32, 32, 48])

    def test_conformer_encoder_block_complexity(self):
        model_config = {
            'n_head': 4,
            'multiplier': 4,
            'key_dim': 36,
            'kernel_size': 32,
            'pos_mode': 'relative',
            'use_bias': False
        }
        self.complexity_test(conformer_encoder_block_complexity,
                             conformer_encoder_block,
                             model_config,
                             [100, 72])

    def test_attention_block_complexity(self):
        model_config = {
            'key_dim': 16,
            'n_head': 4,
            'kernel_size': 3,
            'ff_kernel_size': 3,
            'ff_multiplier': 2,
            'ff_factor0': 1,
            'ff_factor1': 0,
            'abs_pos_encoding': False,
            'use_glu': False,
        }
        self.complexity_test(attention_block_complexity,
                             attention_block,
                             model_config,
                             [20, 32])

        model_config = {
            'key_dim': 16,
            'n_head': 4,
            'kernel_size': 0,
            'ff_kernel_size': 3,
            'ff_multiplier': 0.5,
            'ff_factor0': 0,
            'ff_factor1': 1,
            'abs_pos_encoding': True,
            'use_glu': True,
        }
        self.complexity_test(attention_block_complexity,
                             attention_block,
                             model_config,
                             [20, 32])

        model_config = {
            'depth': 3,
            'key_dim': 16,
            'n_head': 4,
            'kernel_size': 2,
            'ff_kernel_size': 0,
            'ff_multiplier': 0,
            'ff_factor0': 0,
            'ff_factor1': 0,
            'abs_pos_encoding': True,
            'use_glu': True,
        }
        self.complexity_test(attention_block_complexity,
                             attention_block,
                             model_config,
                             [20, 32])

    def test_conv1d_complexity(self):
        target_cx = {'flops': 2304, 'params': 88}
        target_shape = [32, 16]

        self.assertEqual(
            conv1d_complexity(input_shape=[32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1,
                              groups=2,),
            (target_cx, target_shape))
        self.assertEqual(
            conv1d_complexity(input_shape=[32, 3],
                              filters=16,
                              kernel_size=3,
                              strides=1,
                              groups=2,
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

    def test_gru_complexity(self):
        target_cx = {'flops': 954000, 'params': 9360}
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

        input_shape = [8, 32]
        units = 32

        for use_bias in [True, False]:
            inputs = tf.keras.layers.Input(input_shape)
            outputs = tf.keras.layers.GRU(units, use_bias=use_bias,
                                          return_sequences=True)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            cx, output_shape = gru_complexity(
                input_shape=input_shape, units=units, use_bias=use_bias, bi=False)

            self.assertEquals(cx['params'], 
                              sum([K.count_params(p) 
                                   for p in model.trainable_weights]))
            self.assertEquals(tuple(output_shape), model.output_shape[1:])

    def test_lstm_complexity(self):
        input_shape = [8, 32]
        units = 32

        for use_bias in [True, False]:
            inputs = tf.keras.layers.Input(input_shape)
            outputs = tf.keras.layers.LSTM(units, use_bias=use_bias,
                                           return_sequences=True)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            cx, output_shape = lstm_complexity(
                input_shape=input_shape, units=units, use_bias=use_bias, bi=False)

            self.assertEquals(cx['params'], 
                              sum([K.count_params(p) 
                                   for p in model.trainable_weights]))
            self.assertEquals(tuple(output_shape), model.output_shape[1:])

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
    
    def complexity_test(self, 
                        complexity_fn,
                        block_fn,
                        model_config: dict,
                        exp_input_shape: list,
                        verbose=False):
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
        if verbose:
            model.summary()

        cx, output_shape = complexity_fn(model_config, exp_input_shape)

        # TODO: count ops
        self.assertEquals(cx['params'], 
                          sum([K.count_params(p) 
                               for p in model.trainable_weights]))
        self.assertEquals(tuple(output_shape), model.output_shape[1:])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
    tf.test.main()

