import os
import numpy as np
import tensorflow as tf
from layers import *


class LayersTest(tf.test.TestCase):
    def test_conv2d_bn(self):
        layer_args = dict(
            filters=8,
            kernel_size=3,
            activation='relu',
            bn_args={'momentum': 0.99, 'epsilon': 0.001},
        )

        exp_input_shape = 32, 16, 16, 3 
        exp_output_shape = 32, 16, 16, 8

        self.layer_test(conv2d_bn, 
                        layer_args, 
                        exp_input_shape,
                        exp_output_shape)

    def layer_test(self, 
                   layer_fn,
                   layer_args: dict,
                   exp_input_shape, 
                   exp_output_shape):
        '''
        layer_fn: a func that will generate the layer
        layer_args: args for layer_fn
        exp_input_shape: expected input shape
        exp_output_shape: expected output_shape

        "batch size" must be included in both arguments
        ex) [batch, time, chan]
        '''
        x = tf.zeros(exp_input_shape)
        layer = layer_fn(**layer_args)

        y = layer(x)

        self.assertAllEqual(y.shape, exp_output_shape)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

