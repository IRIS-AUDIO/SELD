import os
import numpy as np
import tensorflow as tf
from modules import *


class ModulesTest(tf.test.TestCase):
    def test_mother_stage(self):
        model_config = {
            'depth': 2,
            'filters0': 3,
            'filters1': 3,
            'filters2': 6,
            'kernel_size0': 1,
            'kernel_size1': 3,
            'kernel_size2': 1,
            'connect0': [0],
            'connect1': [0, 0],
            'connect2': [1, 0, 0],
            'strides': [2, 2],
            'activation': 'relu',
        }

        exp_input_shape = 32, 32, 32, 3
        exp_output_shape = 32, 16, 16, 6

        self.block_test(mother_stage, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_bidirectional_GRU_stage(self):
        model_config = {
            'depth': 3,
            'units': 128,
            'dropout_rate': 0.3,
        }
        exp_input_shape = 32, 10, 32, 8
        exp_output_shape = 32, 10, 128

        self.block_test(bidirectional_GRU_stage, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_RNN_stage(self):
        model_config = {
            'depth': 3,
            'units': 64,
            'bidirectional': True,
            'merge_mode': 'ave',
            'rnn_type': 'GRU',
            'dropout_rate': 0.3,
        }
        exp_input_shape = 32, 10, 128
        exp_output_shape = 32, 10, 64

        self.block_test(RNN_stage, model_config, 
                        exp_input_shape, exp_output_shape)

        model_config = {
            'depth': 3,
            'units': 64,
            'bidirectional': True,
            'merge_mode': 'concat',
            'rnn_type': 'LSTM',
            'dropout_rate': 0.,
        }
        exp_input_shape = 32, 10, 128
        exp_output_shape = 32, 10, 128

        self.block_test(RNN_stage, model_config, 
                        exp_input_shape, exp_output_shape)

    def test_simple_dense_stage(self):
        model_config = {
            'depth': 2,
            'units': 128,
            'kernel_size': 3,
            'dense_activation': 'relu',
            'dropout_rate': 0,
            'kernel_regularizer': {'l1': 0, 'l2': 1e-3},
        }
        exp_input_shape = 32, 10, 64
        exp_output_shape = 32, 10, 128

        self.block_test(simple_dense_stage, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_transformer_encoder_stage(self):
        model_config = {
            'depth': 3,
            'n_head': 8, # mandatory
            'key_dim': 8,
            'ff_multiplier': 128, # mandatory
            'kernel_size': 1, # mandatory
            'dropout_rate': 0.1,
        }

        exp_input_shape = 32, 20, 64
        exp_output_shape = 32, 20, 64

        self.block_test(transformer_encoder_stage, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)


    def test_conformer_encoder_stage(self):
        model_config = {
            'depth': 3,
            'key_dim': 36, # mandatory
            'n_head' : 4,
            'kernel_size' : 32,
            'activation': 'swish',
            'dropout_rate': 0,
        }
        
        exp_input_shape = 32, 100, 64 # batch, time, feat
        exp_output_shape = 32, 100, 64 # batch, time, feat
    
        self.block_test(conformer_encoder_stage, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_mother_block(self):
        model_config = {
            'filters0': 6,
            'filters1': 8,
            'filters2': 0,
            'kernel_size0': 3,
            'kernel_size1': 3,
            'kernel_size2': 0,
            'connect0': [0],
            'connect1': [0, 1],
            'connect2': [1, 0, 1],
            'strides': [1, 2],
            'activation': 'relu',
        }

        exp_input_shape = 32, 32, 32, 3
        exp_output_shape = 32, 32, 16, 11

        self.block_test(mother_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

        # with SE
        model_config = {
            'filters0': 6,
            'filters1': 8,
            'filters2': 0,
            'kernel_size0': 3,
            'kernel_size1': 3,
            'kernel_size2': 0,
            'connect0': [0],
            'connect1': [0, 1],
            'connect2': [1, 0, 1],
            'strides': [1, 2],
            'activation': 'relu',
            'squeeze_ratio': 0.5,
            'se_activation': 'swish',
        }

        exp_input_shape = 32, 32, 32, 3
        exp_output_shape = 32, 32, 16, 11

        self.block_test(mother_block, 
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

    def test_RNN_block(self):
        model_config = {
            'units': 64,
            'bidirectional': False,
            'merge_mode': None,
            'rnn_type': 'GRU',
            'dropout_rate': 0.3,
        }

        # 1D inputs
        exp_input_shape = 32, 10, 32
        exp_output_shape = 32, 10, 64

        self.block_test(RNN_block, 
                        model_config, 
                        exp_input_shape,
                        exp_output_shape)

    def test_simple_dense_block(self):
        model_config = {
            'units': [128, 128], # mandatory
            'kernel_size': 5,
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

    def test_transformer_encoder_block(self):
        model_config = {
            'n_head': 8, # mandatory
            'key_dim': 8,
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

    def test_conformer_encoder_block(self):
        model_config = {
            'key_dim': 36, # mandatory
            'n_head' : 4,
            'kernel_size' : 32,
            'activation': 'swish',
            'dropout_rate': 0,
            'pos_mode':'relative'
        }
        
        exp_input_shape = 32, 100, 64 # batch, time, feat
        exp_output_shape = 32, 100, 64 # batch, time, feat
    
        self.block_test(conformer_encoder_block, 
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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
    tf.test.main()

