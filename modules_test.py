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

    def test_res_bottleneck_stage(self):
         model_config = {
             'depth': 2, # mandatory
             'filters': 32, # mandatory (for res bottleneck block)
             'strides': 2, # mandatory (for res bottleneck block)
             'groups': 2, # mandatory (for res bottleneck block)
             'bottleneck_ratio': 2, # mandatory (for res bottleneck block)
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
             'groups': 2, # mandatory
             'bottleneck_ratio': 2, # mandatory
         }

         exp_input_shape = 32, 32, 32, 3
         exp_output_shape = 32, 16, 16, 32

         self.block_test(res_bottleneck_block, 
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
        exp_output_shape = 32, 60, 8192

        self.block_test(xception_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_dense_net_block(self):
        model_config = {
            'filters' : 32,
            'name': 'dense_net_block',
            'block_num': [6,12,24,16],
            'kernel_regularizer': {'l1': 1e-3, 'l2': 0.}
        }

        exp_input_shape = 2, 300, 64, 3
        exp_output_shape = 2, 60, 2040

        self.block_test(dense_net_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_resnet50_block(self):
        model_config = {
            'filters': 32,
            'name': 'resnet50_block',
            'block_num': [3, 4, 6, 3],
            'kernel_regularizer': {
                'l1': 0,
                'l2': 1e-3
            }
        }

        exp_input_shape = 2, 300, 64, 3
        exp_output_shape = 2, 60, 2048

        self.block_test(resnet50_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_bidirectional_GRU_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'dropout_rate': 0.3,
        }

        exp_input_shape = 32, 10, 32, 8
        exp_output_shape = 32, 10, 128

        self.block_test(bidirectional_GRU_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_transformer_encoder_layer(self):
        model_config = {
            'd_model': 64, # mandatory
            'n_head': 8, # mandatory
            'dim_feedforward': 128,
            'dropout_rate': 0.1,
        }

        exp_input_shape = 32, 20, 64
        exp_output_shape = 32, 20, 64

        self.block_test(transformer_encoder_layer, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_simple_dense_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'n_classes': 10, # mandatory 
            'name': 'simple_dense_block',
            'activation': 'relu',
            'dropout_rate': 0,
            'kernel_regularizer': {'l1': 0, 'l2': 1e-3},
        }

        exp_input_shape = 32, 10, 128 # batch, time, feat
        exp_output_shape = 32, 10, model_config['n_classes'] # batch, time, feat

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
    tf.test.main()

