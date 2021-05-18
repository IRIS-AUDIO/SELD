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
        input_shape = [7, 80, 1] # win_size, n_mels
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
        vad = vad_architecture(input_shape, model_config)
        self.assertEqual(vad.output_shape[1:], (7,))

        model_config = {
            'flatten': False,
            'last_unit': 1,
            'BLOCK0': 'simple_dense_stage',
            'BLOCK0_ARGS': {
                'depth': 2,
                'units': 512,
                'dense_activation': 'relu',
                'dropout_rate': 0.5,
            }
        }
        vad = vad_architecture(input_shape, model_config)
        self.assertEqual(vad.output_shape[1:], (7,))

    def test_spectro_temporal_attention_based_VAD(self):
        input_shape = [7, 80, 1] # win_size, n_mels
        model_config = {}
        vad = spectro_temporal_attention_based_VAD(input_shape, model_config)

        self.assertEqual(vad.output_shape[0][1:], (7, 1))
        self.assertEqual(vad.output_shape[1][1:], (7, 1))
        self.assertEqual(vad.output_shape[2][1:], (7,))

        # test its trainability
        x = tf.random.uniform([32, 7, 80, 1])
        y = tf.random.uniform([32, 7], maxval=2, dtype=tf.int32)

        vad.compile('adam', loss=tf.keras.losses.BinaryCrossentropy())
        vad.fit(x, y)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

