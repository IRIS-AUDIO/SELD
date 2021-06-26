import os
import tensorflow as tf
import tensorflow.keras.backend as K
from modules import *
from stage_complexity import *


class StageComplexityTest(tf.test.TestCase):
    def test_mother_stage_complexity(self):
        model_config = {
            "depth": 2,
            "filters0": 64,
            "filters1": 64,
            "filters2": 0,
            "kernel_size0": 3,
            "kernel_size1": 1,
            "kernel_size2": 0,
            "connect0": [
                1
            ],
            "connect1": [
                1,
                1
            ],
            "connect2": [
                1,
                0,
                1
            ],
            "strides": [
                1,
                2
            ]
        }
        self.complexity_test(mother_stage_complexity,
                             mother_stage,
                             model_config,
                             [32, 32, 16])

    def test_bidirectional_GRU_stage_complexity(self):
        model_config = {
            'depth': 3,
            'units': 128,
        }
        self.complexity_test(bidirectional_GRU_stage_complexity,
                             bidirectional_GRU_stage,
                             model_config,
                             [32, 32, 16])

    def test_simple_dense_stage_complexity(self):
        model_config = {
            'depth': 2,
            'units': 128,
            'kernel_size': 3,
        }
        self.complexity_test(simple_dense_stage_complexity,
                             simple_dense_stage,
                             model_config,
                             [32, 32, 16])

    def test_transformer_encoder_stage_complexity(self):
        model_config = {
            'depth': 3,
            'n_head': 8,
            'key_dim': 8,
            'ff_multiplier': 128,
            'kernel_size': 3,
            'dropout_rate': 0.1,
        }
        self.complexity_test(transformer_encoder_stage_complexity,
                             transformer_encoder_stage,
                             model_config,
                             [32, 32, 16])

    def test_conformer_encoder_stage_complexity(self):
        model_config = {
            'depth': 3,
            'key_dim': 36, # mandatory
            'n_head' : 4,
            'kernel_size' : 32,
        }
        self.complexity_test(conformer_encoder_stage_complexity,
                             conformer_encoder_stage,
                             model_config,
                             [32, 32, 16])

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

