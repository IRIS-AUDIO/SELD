import os
import numpy as np
import tensorflow as tf
from layers import *


class LayersTest(tf.test.TestCase):
    def test_simple_conv_block(self):
        model_config = {
            'filters': [128, 128], # mandatory
            'pool_size': [[4, 4], [1, 1]]
        }

        exp_input_shape = 32, 32, 32, 3
        exp_output_shape = 32, 8, 8, 128

        self.block_test(simple_conv_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_bidirectional_GRU_block(self):
        model_config = {
            'units': [128, 128], # mandatory
        }

        exp_input_shape = 32, 10, 32, 8
        exp_output_shape = 32, 10, 128

        self.block_test(bidirectional_GRU_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_simple_dense_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'n_classes': 10, # mandatory 
            'dropout_rate': 0,
            'activation': 'relu',
            'name': 'simple_dense_block'
        }

        exp_input_shape = 32, 10, 128 # batch, time, feat
        exp_output_shape = 32, 10, model_config['n_classes'] # batch, time, feat

        self.block_test(simple_dense_block, 
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

