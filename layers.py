import tensorflow as tf
from tensorflow.keras.layers import *

"""
Layers

This is only for implementing layers.
You should not import class or functions from modules or models
"""


class Routing(Layer):
    def __init__(self, out_channels, dropout_rate, temperature=30, **kwargs):
        super(Routing, self).__init__(**kwargs)
        self.avgpool = GlobalAveragePooling2D()
        self.dropout = Dropout(rate=dropout_rate)
        self.fc = Dense(units=out_channels)
        self.softmax = Softmax()
        self.temperature = temperature

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fc': self.fc1
        })
        return config

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, c, h, w)
        :return: (b, out_features)
        """
        out = self.avgpool(inputs)
        out = self.dropout(out)

        # refer to paper: https://arxiv.org/pdf/1912.03458.pdf
        out = self.softmax(self.fc(out) * 1.0 / self.temperature)
        return out


class CondConv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, use_bias=True, num_experts=3, padding="same", **kwargs):
        super(CondConv2D, self).__init__(**kwargs)

        self.routing = Routing(out_channels=num_experts, dropout_rate=0.2, name="routing_layer")
        self.convs = []
        for _ in range(num_experts):
            self.convs.append(conv2d(filters=filters, stride=stride, kernel_size=kernel_size, use_bias=use_bias, padding=padding))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'convs': self.convs
        })
        return config

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, h, w, c)
        :return: (b, h_out, w_out, filters)
        """
        routing_weights = self.routing(inputs)
        feature = routing_weights[:, 0] * tf.transpose(self.convs[0](inputs), perm=[1, 2, 3, 0])
        for i in range(1, len(self.convs)):
            feature += routing_weights[:, i] * tf.transpose(self.convs[i](inputs), perm=[1, 2, 3, 0])
        feature = tf.transpose(feature, perm=[3, 0, 1, 2])
        return feature


class DConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), use_bias=True, padding='same', activation='softmax'):
        super(DConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size if type(kernel_size) in (list, tuple) else (int(kernel_size), int(kernel_size))
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                    shape=[
                                    self.kernel_size[0], 
                                    self.kernel_size[1], 
                                    input_shape[-1], 
                                    self.filters], trainable=True)
        self.bias = self.add_weight("bias",
                        shape=[self.filters], trainable=True)
        self.fc1 = Dense(input_shape[-1], use_bias=self.use_bias)
        self.fc2 = Dense(input_shape[-1], use_bias=self.use_bias)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fc1': self.fc1,
            'fc2': self.fc2
        })
        return config

    def call(self, input):
        k = GlobalAveragePooling2D()(input)
        k = self.fc1(k)
        k = ReLU()(k)
        k = self.fc2(k)
        k = Reshape((1,1,-1,1))(k)
        weighted_kernel = self.kernel * k # (batch, h, w, input_channel, output_channel)
        x = tf.map_fn(self._conv, (input, weighted_kernel), dtype=input.dtype)
        if self.use_bias:
            x = x + self.bias
        x = Activation(self.activation)(x)
        return x
    
    def _conv(self, input):
        # input[0]: (time, freq, channel), input[1]: kernel(h, w, input_channel, output_channel)
        return tf.squeeze(tf.keras.backend.conv2d(x=input[0][tf.newaxis,...], 
                    kernel=input[1], strides=self.strides, padding=self.padding), 0)

