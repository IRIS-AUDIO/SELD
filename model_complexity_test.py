import os
import tensorflow as tf
import tensorflow.keras.backend as K
from models import *
from modules import *
from model_complexity import *


class ModelComplexityTest(tf.test.TestCase):
    def test_conv_temporal_complexity(self):
        input_shape = [30, 256, 3]
        model_config = {
            "BLOCK0": "res_bottleneck_stage",
            "BLOCK0_ARGS": {
                "filters": 32,
                "depth": 3,
                "strides": [1, 2]
            },
            "SED": "simple_dense_stage",
            "SED_ARGS": {
                "units": 128,
                "depth": 1,
                "n_classes": 14,
                "activation": "sigmoid",
                "name": "sed_out"
            },
            "DOA": "simple_dense_stage",
            "DOA_ARGS": {
                "units": 128,
                "depth": 1,
                "n_classes": 42,
                "activation": "tanh",
                "name": "doa_out"
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

