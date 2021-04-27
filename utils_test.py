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

    def test_safe_tuple(self):
        self.assertEqual((1, 1), safe_tuple(1, 2))
        self.assertEqual((1, 3), safe_tuple((1, 3), 2))
        with self.assertRaises(ValueError):
            safe_tuple((1, 2, 3), 2)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

