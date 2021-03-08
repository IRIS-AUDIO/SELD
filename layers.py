import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *


def simple_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = Conv2D(filters[i], kernel_size=3, padding='same', 
                       kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def bidirectional_GRU_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']

    dropout_rate = model_config.get('dropout_rate', 0.)

    def GRU_block(inputs):
        x = inputs
        if len(x.shape) == 4: # [batch, time, freq, chan]
            x = Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

        for units in units_per_layer:
            x = Bidirectional(
                GRU(units, activation='tanh', 
                    dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)
        return x

    return GRU_block


def simple_dense_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['units']
    n_classes = model_config['n_classes']

    name = model_config.get('name', None)
    activation = model_config.get('activation', None)
    dropout_rate = model_config.get('dropout_rate', 0)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def dense_block(inputs):
        x = inputs
        for units in units_per_layer:
            x = TimeDistributed(
                Dense(units, kernel_regularizer=kernel_regularizer))(x)
            x = Dropout(dropout_rate)(x)
        x = TimeDistributed(
            Dense(n_classes, activation=activation, name=name,
                  kernel_regularizer=kernel_regularizer))(x) 
        return x

    return dense_block


def dynamic_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']
    dropout_rate = model_config.get('dropout_rate', 0.)
    activation = model_config.get('activation', 'softmax')    
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = DConv2D(filters[i], kernel_size=3, padding='same', activation=activation)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def cond_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']
    dropout_rate = model_config.get('dropout_rate', 0.)  
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = CondConv2D(filters[i], kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x

    return conv_block


def conv2d(kernel_size, stride, filters, kernel_regularizer=tf.keras.regularizers.l2(1e-3), padding="same", use_bias=False,
           kernel_initializer="he_normal", **kwargs):
    return Conv2D(kernel_size=kernel_size, strides=stride, filters=filters, kernel_regularizer=kernel_regularizer, padding=padding,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, **kwargs)


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


def xception_block(model_config: dict):
    filters = model_config['filters']
    block_num = model_config['block_num']
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def _xception_block(inputs):
        x = Conv2D(filters, 3, use_bias=True, name='block1_conv2', padding='same', kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)
        x = MaxPooling2D(pool_size=(5,1))(x)

        for _ in range(block_num):
            residual = Conv2D(x.shape[-1] * 2, (1,1), strides=(1,1), padding='same', use_bias=True, kernel_regularizer=kernel_regularizer)(x)
            residual = BatchNormalization()(residual)

            x = SeparableConv2D(
                x.shape[-1] * 2, 3, padding='same', use_bias=True)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(
                x.shape[-1], 3, padding='same', use_bias=True)(x)
            x = BatchNormalization()(x)

            x = add([x, residual])


        return x
    return _xception_block
    