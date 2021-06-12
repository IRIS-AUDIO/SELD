import os
import tensorflow as tf
import tensorflow.keras.backend as K
from models import *
from model_complexity import *


class ModelComplexityTest(tf.test.TestCase):
    def test_conv_temporal_complexity(self):
        input_shape = [30, 256, 3]
        model_config = {
            "n_classes": 12,
            "first_pool_size": [5, 2],
            "BLOCK0": "mother_stage",
            "BLOCK0_ARGS": {
                "depth": 2,
                "filters0": 0,
                "filters1": 96,
                "filters2": 0,
                "kernel_size0": 0,
                "kernel_size1": 3,
                "kernel_size2": 0,
                "connect0": [
                    1
                ],
                "connect1": [
                    1,
                    0
                ],
                "connect2": [
                    1,
                    0,
                    1
                ],
                "strides": [
                    1,
                    3
                ]
            },
            "BLOCK1": "simple_dense_stage",
            "BLOCK1_ARGS": {
                "depth": 1,
                "units": 192,
                "dense_activation": "relu",
                "dropout_rate": 0.0
            },
            "BLOCK2": "conformer_encoder_stage",
            "BLOCK2_ARGS": {
                "depth": 2,
                "key_dim": 24,
                "n_head": 4,
                "kernel_size": 24,
                "multiplier": 2,
                "pos_encoding": None
            },
            "SED": "conformer_encoder_stage",
            "SED_ARGS": {
                "depth": 1,
                "key_dim": 48,
                "n_head": 4,
                "kernel_size": 8,
                "multiplier": 2,
                "pos_encoding": None
            },
            "DOA": "bidirectional_GRU_stage",
            "DOA_ARGS": {
                "depth": 2,
                "units": 128
            }
        }
        self.model_complexity_test(conv_temporal_complexity,
                                   conv_temporal,
                                   model_config,
                                   input_shape)

    def test_vad_architecture_complexity(self):
        input_shape = [7, 80, 1]
        model_config = {
            'flatten': True,
            'last_unit': 7,
            'BLOCK0': 'simple_dense_stage',
            'BLOCK0_ARGS': {
                'depth': 2,
                'units': 512,
                'dense_activation': 'relu',
                'dropout_rate': 0.5,
            }
        }
        self.model_complexity_test(vad_architecture_complexity,
                                   vad_architecture,
                                   model_config,
                                   input_shape)

    def model_complexity_test(self, 
                              complexity_fn,
                              model_fn,
                              model_config: dict,
                              input_shape: list,
                              verbose=False):
        model = model_fn(input_shape, model_config)
        if verbose:
            model.summary()

        cx, output_shape = complexity_fn(model_config, input_shape)

        self.assertEquals(cx['params'], 
                          sum([K.count_params(p) 
                               for p in model.trainable_weights]))

        if isinstance(model.output_shape, list):
            for i in range(len(model.output_shape)):
                self.assertEquals(tuple(output_shape[i]), 
                                  model.output_shape[i][1:])
        else:
            self.assertEquals(tuple(output_shape), 
                              model.output_shape[1:])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
    tf.test.main()

