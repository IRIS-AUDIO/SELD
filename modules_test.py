import os
import numpy as np
import tensorflow as tf
from modules import *


class ModulesTest(tf.test.TestCase):
    def test_simple_conv_block(self):
        model_config = {
            'filters': [128, 128], # mandatory
            'pool_size': [[4, 4], [1, 1]], # mandatory
            'dropout_rate': 0.3,
            'kernel_regularizer': {'l1': 1e-3, 'l2': 0.},
        }

        exp_input_shape = 32, 32, 32, 3
        exp_output_shape = 32, 8, 8, 128

        self.block_test(simple_conv_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_another_conv_block(self):
         model_config = {
             'depth': 2, # mandatory
             'filters': 32, # mandatory
             'pool_size': 2, # mandatory 
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(another_conv_block,
                         model_config,
                         exp_input_shape,
                         exp_output_shape)

    def test_res_basic_stage(self):
         model_config = {
             'depth': 2, # mandatory
             'filters': 32, # mandatory (for res basic block)
             'strides': 2, # mandatory (for res basic block)
             'groups': 1, 
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(res_basic_stage,
                         model_config,
                         exp_input_shape,
                         exp_output_shape)

    def test_res_basic_block(self):
         model_config = {
             'filters': 32, # mandatory
             'strides': 2, # mandatory
             'groups': 1,
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(res_basic_block, 
                         model_config, 
                         exp_input_shape,
                         exp_output_shape)

    def test_res_bottleneck_stage(self):
         model_config = {
             'depth': 2, # mandatory
             'filters': 32, # mandatory (for res bottleneck block)
             'strides': 2, # mandatory (for res bottleneck block)
             'groups': 1, 
             'bottleneck_ratio': 2,
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(res_bottleneck_stage,
                         model_config,
                         exp_input_shape,
                         exp_output_shape)

    def test_res_bottleneck_block(self):
         model_config = {
             'filters': 32, # mandatory
             'strides': 2, # mandatory
             'groups': 1,
             'bottleneck_ratio': 2,
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(res_bottleneck_block, 
                         model_config, 
                         exp_input_shape,
                         exp_output_shape)

    def test_dense_net_block(self):
        model_config = {
            'growth_rate' : 6, # mandatory
            'depth': 4, # mandatory
            'strides': [2, 2], # mandatory
            'bottleneck_ratio': 4,
            'reduction_ratio': 0.5,
        }

        exp_input_shape = 2, 32, 32, 6
        exp_output_shape = 2, 16, 16, 15

        self.block_test(dense_net_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_sepformer_block(self):
        model_config = {
            'n_head': 4, # mandatory
            'key_dim': 4, # mandatory
            'ff_multiplier': 2, # mandatory
            'kernel_size': 3, # mandatory
            'pos_encoding': 'basic',
        }

        exp_input_shape = 32, 50, 64, 12
        exp_output_shape = 32, 50, 64, 12

        self.block_test(sepformer_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_xception_block(self):
        model_config = {
            'filters' : 32,
            'name': 'xception_block',
            'block_num': 8,
            'kernel_regularizer': {'l1': 1e-3, 'l2': 0.}
        }

        exp_input_shape = 32, 300, 64, 3
        exp_output_shape = 32, 60, 4, 2048

        self.block_test(xception_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_xception_basic_block(self):
        model_config = {
            'filters' : 32, # mandatory
            'mid_ratio': 1,
            'strides': (1, 2),
            'kernel_regularizer': {'l1': 1e-3, 'l2': 0.}
        }

        exp_input_shape = 32, 300, 64, 3
        exp_output_shape = 32, 300, 32, 32

        self.block_test(xception_basic_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_bidirectional_GRU_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'dropout_rate': 0.3,
        }

        # 1D inputs
        exp_input_shape = 32, 10, 32
        exp_output_shape = 32, 10, 128

        self.block_test(bidirectional_GRU_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

        # 2D inputs
        exp_input_shape = 32, 10, 32, 8
        exp_output_shape = 32, 10, 128

        self.block_test(bidirectional_GRU_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_transformer_encoder_block(self):
        model_config = {
            'n_head': 8, # mandatory
            'key_dim': 4, # mandatory
            'ff_multiplier': 128, # mandatory
            'kernel_size': 1, # mandatory
            'dropout_rate': 0.1,
        }

        exp_input_shape = 32, 20, 64
        exp_output_shape = 32, 20, 64

        self.block_test(transformer_encoder_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_simple_dense_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'dense_activation': 'relu',
            'dropout_rate': 0,
            'kernel_regularizer': {'l1': 0, 'l2': 1e-3},
        }
        exp_input_shape = 32, 10, 64
        exp_output_shape = 32, 10, 128

        self.block_test(simple_dense_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_identity_block(self):
        model_config = {}

        exp_input_shape = 32, 16, 16, 8
        exp_output_shape = 32, 16, 16, 8

        self.block_test(identity_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_conformer_encoder_block(self):
        model_config = {
            'key_dim': 36, # mandatory
            'n_head' : 4,
            'kernel_size' : 32,
            'activation': 'swish',
            'dropout_rate': 0,
        }
        
        exp_input_shape = 32, 100, 64 # batch, time, feat
        exp_output_shape = 32, 100, 64 # batch, time, feat
    
        self.block_test(conformer_encoder_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def block_test(self, 
                   block_fn,
                   model_config: dict,
                   exp_input_shape, 
                   exp_output_shape):
        '''
        block_fn: a func that will generate the block
        model_config: model_config for block_fn
        exp_input_shape: expected input shape
        exp_output_shape: expected output_shape
        "batch size" must be included in both arguments
        ex) [batch, time, chan]
        '''
        inputs = tf.keras.layers.Input(exp_input_shape[1:])
        outputs = block_fn(model_config)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        x = tf.zeros(exp_input_shape)
        y = model(x)

        self.assertAllEqual(y.shape, exp_output_shape)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
    tf.test.main()
