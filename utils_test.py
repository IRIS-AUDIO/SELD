import os
import tensorflow as tf
from utils import *


class UtilsTest(tf.test.TestCase):
    def test_safe_div(self):
        self.assertFalse(tf.math.is_nan(safe_div(1, 0.)))

    def test_dict_add(self):
        a = {'a': 3}
        b = {'a': 2, 'b': 4}

        gt = {'a': 5, 'b': 4}

        c = dict_add(a, b)
        self.assertEqual(c, gt)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

