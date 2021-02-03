import os
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def setUp(self):
        self.batch = 32
        self.freq = 257
        self.time = 10
        self.chan = 2
        self.n_classes = 2

        self.x = tf.zeros((self.batch, self.freq, self.time, self.chan))

    def test_build_seldnet(self):
        model = build_seldnet([self.freq, self.time, self.chan],
                              self.n_classes)
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

