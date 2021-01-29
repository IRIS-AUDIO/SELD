import os
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def test_build_seldnet(self):
        batch, freq, time, chan = 32, 257, 10, 2
        n_classes = 2
        x = tf.zeros((batch, freq, time, chan))

        model = build_seldnet([freq, time, chan], n_classes)
        y = model.predict(x)

        # SED
        self.assertAllEqual(y[0].shape, [batch, time, n_classes])
        
        # DOA
        self.assertAllEqual(y[1].shape, [batch, time, n_classes*3])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

