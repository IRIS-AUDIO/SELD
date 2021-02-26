import os
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def setUp(self):
        self.batch = 32
        self.freq = 257
        self.time = 60
        self.chan = 2
        self.n_classes = 2

        self.x = tf.zeros((self.batch, self.freq, self.time, self.chan))

    def test_seldnet_architecture(self):
        model = seldnet_architecture(
            self.x.shape,
            hlfr=lambda x: x,
            tcr=lambda x: tf.reduce_mean(x, axis=1),
            sed=lambda x: tf.keras.layers.Dense(self.n_classes)(x),
            doa=lambda x: tf.keras.layers.Dense(self.n_classes*3)(x))
        y = model.predict(self.x)

        # SED
        self.assertAllEqual(y[0].shape, 
                            [self.batch, self.time, self.n_classes])
        # DOA
        self.assertAllEqual(y[1].shape, 
                            [self.batch, self.time, self.n_classes*3])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

