import os
import tensorflow as tf
from utils import *


class UtilsTest(tf.test.TestCase):
    def test_safe_div(self):
        self.assertFalse(tf.math.is_nan(safe_div(1, 0.)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

