import os
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Flatten, Input, BatchNormalization, Activation, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv1D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
import keras.backend as K

filters = 64

init = initializers.glorot_uniform(seed=717)


def build_model(input_layer, filters, init, kernel):

    pad = 'same' # 'valid'

    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(input_layer)
    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2,1))(x)

    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(4,1))(x)

    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = Conv2D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(4,1))(x)

    x = Conv2D(filters=2, kernel_size=(2,2), strides= (1,1), activation="relu", padding='valid', kernel_initializer=init)(x)

    x = AveragePooling2D(pool_size=(K.int_shape(x)[-3],1))(x)

    x = Activation(("softmax"))(x)

    output_layer = Flatten()(x)

    return output_layer

# Model


def cnn_2D(n_timesteps, n_features, kernel):

    input_layer = Input((n_timesteps, n_features,1))
    output_layer = build_model(input_layer, filters, init, kernel=kernel)

    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 

    return(model)

