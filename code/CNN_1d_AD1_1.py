import os
from keras.models import Sequential
from keras.layers import Flatten, Input, BatchNormalization, Activation, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv1D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras import initializers
import keras.backend as K
import numpy as np
from keras.utils import np_utils
import random
import pdb

filters = 64
init = initializers.glorot_uniform(seed=717)



def build_model(input_layer, filters, init, kernel):

    pad = 'valid'

    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(input_layer)
    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)

    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)

    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=4)(x)

    x = Conv1D(filters=2, kernel_size=2, strides= 1, activation="relu", padding='valid', kernel_initializer=init)(x)

    x = AveragePooling1D(pool_size=(K.int_shape(x)[-2]))(x)

    x = Activation(("softmax"))(x)

    output_layer = Flatten()(x)

    return output_layer

# Model


def cnn_1d(n_timesteps, n_features, kernel):

    input_layer = Input((n_timesteps, n_features))
    output_layer = build_model(input_layer, filters, init, kernel=kernel)

    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return(model)



