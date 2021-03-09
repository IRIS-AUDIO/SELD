import os
import tensorflow as tf
from models import *


class ModelsTest(tf.test.TestCase):
    def test_seldnet(self):
        raise NotImplemented()

    def test_seldnet_v1(self):
        raise NotImplemented()

    def test_seldnet_v1(self):
        raise NotImplemented()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.test.main()

