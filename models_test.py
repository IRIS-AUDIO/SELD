import os
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def test_seldnet(self):
        raise NotImplemented()

    def test_seldnet_v1(self):
        raise NotImplemented()

    def test_conv_temporal(self):
        raise NotImplemented()

    def test_vad_architecture(self):
        input_shape = [7, 80] # win_size, n_mels
        model_config = {
            'flatten': True,
            'last_unit': 7,
            'BLOCK0': 'simple_dense_block',
            'BLOCK0_ARGS': {
                'units': [512, 512],
                'dense_activation': 'relu',
                'dropout_rate': 0.5,
            }
        }
        vad = vad_architecture(input_shape, model_config)
        vad.summary()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

