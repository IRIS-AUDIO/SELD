import os
import tensorflow as tf
import tensorflow.keras.backend as K
from modules import *
from stage_complexity import *


class StageComplexityTest(tf.test.TestCase):
    def test_simple_conv_stage_complexity(self):
        model_config = {
            'filters': 24,
            'depth': 4,
            'pool_size': (2, 3),
        }
        self.complexity_test(simple_conv_stage_complexity,
                             simple_conv_stage,
                             model_config,
                             [32, 32, 16])

    def test_another_conv_stage_complexity(self):
        model_config = {
            'filters': 24,
            'depth': 4,
            'pool_size': (2, 3),
        }
        self.complexity_test(another_conv_stage_complexity,
                             another_conv_stage,
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
    os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
    tf.test.main()

