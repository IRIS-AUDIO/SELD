import os
import numpy as np
import tensorflow as tf
from layers import *


class LayersTest(tf.test.TestCase):
    def test_Routing(self):
        raise NotImplemented()

    def test_CondConv2D(self):
        raise NotImplemented()

    def test_DConv2D(self):
        raise NotImplemented()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

