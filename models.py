import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *

import modules

"""
Models

This defines model structures (architectures)
You should not define modules nor layers here.
"""


def seldnet(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def seldnet_v1(input_shape, model_config):
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    doa *= Concatenate()([sed] * 3)
    doa = tanh(doa) 

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def xception_gru(input_shape, model_config):
    # interprets model_config to an actual model
    inputs = Input(shape=input_shape[-3:])

    x = getattr(modules, model_config.FIRST)(model_config.FIRST_ARGS)(inputs)
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)
    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    return tf.keras.Model(inputs=inputs, outputs=[sed, doa])


def res_identity(x, filters): 
 
  x_skip = x 
  f1, f2 = filters

  
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

 
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  
  x = Add()([x, x_skip])
  x = Activation('relu')(x)

  return x

def res_conv(x, s, filters):
   
    x_skip = x
    f1, f2 = filters

    
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)

   
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x
  
def resnet50(data_in):
    input_im = Input(data_in[1:])
    spec_cnn = ZeroPadding2D(padding=(3, 3))(input_im)


    spec_cnn = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(spec_cnn)
    spec_cnn = BatchNormalization()(spec_cnn)
    spec_cnn = Activation('relu')(spec_cnn)
    spec_cnn = MaxPooling2D((3, 3), strides=(2, 2))(spec_cnn)


    spec_cnn = res_conv(spec_cnn, s=1, filters=(64, 256))
    spec_cnn = res_identity(spec_cnn, filters=(64, 256))
    spec_cnn = res_identity(spec_cnn, filters=(64, 256))

   
    spec_cnn = res_conv(spec_cnn, s=2, filters=(128, 512))
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))


    spec_cnn = res_conv(spec_cnn, s=2, filters=(256, 1024))
    for _ in range(5):
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))

    spec_cnn = res_conv(spec_cnn, s=2, filters=(512, 2048))
    spec_cnn = res_identity(spec_cnn, filters=(512, 2048))
    spec_cnn = res_identity(spec_cnn, filters=(512, 2048))

    spec_cnn =  ZeroPadding2D(padding=(25, 25))(spec_cnn)
    #spec_cnn = AveragePooling2D((2, 2), padding='same')(spec_cnn)

    #spec_cnn = Flatten()(spec_cnn)
    #spec_cnn = Dense(14, activation='softmax', kernel_initializer='he_normal')(spec_cnn)

    model = tf.keras.Model(inputs=input_im, outputs=spec_cnn, name='Resnet50')
    model.summary()
    return model
    
def resnet(input_shape, model_config):
    # model definition
    spec_start = Input(shape=input_shape[-3:])
    
    spec_cnn = resnet50(input_shape)(spec_start)
    # [b, t, f, c] -> [b, t, f*c]
    # (None, 60, a, b)
    #pdb.set_trace()
    #assert spec_cnn.shape[1] == 60 and len(spec_cnn.shape) == 4, 'something wrong'
    x = Reshape((-1, spec_cnn.shape[-2]*spec_cnn.shape[-1]))(spec_cnn)

    # (None, 60, a*b)
    
    x = getattr(modules, model_config.SECOND)(model_config.SECOND_ARGS)(x)

    
    sed = getattr(modules, model_config.SED)(model_config.SED_ARGS)(x)
    doa = getattr(modules, model_config.DOA)(model_config.DOA_ARGS)(x)

    model = tf.keras.Model(inputs=spec_start, outputs=[sed, doa])
    return model

