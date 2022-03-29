import os
os.environ['TF_KERAS'] = '1'
import os
from keras.models import Sequential
#from keras.optimizers import SGD
from keras.layers import Flatten, Input, BatchNormalization, Activation, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv1D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
#from keras_radam import RAdam # Note need os.environ['TF_KERAS'] = '1' above
import keras.backend as K
#import MyCallbacks
import numpy as np
from keras.utils import np_utils
import random
import pdb

filters = 64
filters1 = 32 # 64
filters2 = 32 # 128
filters3 = 32 # 256
# init = 'he_normal'
init = initializers.glorot_uniform(seed=717)
# init = initializers.he_normal(seed=717)


def fe_block(block_input, filters, init):

    pad = 'valid'

    conv1 = Conv2D(filters*1, (3,1), activation='relu', padding=pad, kernel_initializer=init)(block_input)

    conv1 = Conv2D(filters*1, (3,1), activation='relu', padding=pad, kernel_initializer=init)(conv1)

    conv1 = Conv2D(filters * 1, (3, 1), activation=None, padding=pad, kernel_initializer=init)(conv1)

    skip_conv = Cropping2D(cropping=((3, 3), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])


    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def fe_block_k(block_input, filters, init, dm = 1, kernel=3):

    pad = 'valid'
    adj = int((kernel-3)/2)

    conv1 = Conv2D(filters * dm, (kernel, 1), activation='relu', padding=pad, kernel_initializer=init)(block_input)
    conv1 = Conv2D(filters*1, (3,1), activation='relu', padding=pad, kernel_initializer=init)(conv1)

    conv1 = Conv2D(filters * 1, (1, 1), activation=None, padding=pad, kernel_initializer=init)(conv1)

    skip_conv = Cropping2D(cropping=((2+adj, 2+adj), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])


    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def build_model(input_layer, filters, init, kernel):

    pad = 'valid'

    # x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(input_shape=input_layer)
    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(input_layer)
    x = Conv1D(filters*1, kernel_size = kernel, activation='relu', padding=pad, kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    # x = AveragePooling1D(pool_size=2, strides=1)(x)
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

    # x = AveragePooling1D(pool_size=4)(x)
    x = AveragePooling1D(pool_size=(K.int_shape(x)[-2]))(x)
    # x = (AveragePooling2D(pool_size=(8, 1), strides=(1, 1), padding='valid'))(x) #original

    x = Activation(("softmax"))(x)

    output_layer = Flatten()(x)

    return output_layer

# Model


def cnn_1d(n_timesteps, n_features, kernel):

    input_layer = Input((n_timesteps, n_features))
    output_layer = build_model(input_layer, filters, init, kernel=kernel)

    BatchSize = 4096

    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)

    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=RAdam(), metrics=['accuracy'])

    print(model.summary())

    return(model)


def fe_block_dep(block_input, filters, init, dm = 1, kernel=3):

    pad = 'valid'
    adj = int((kernel-3)/2)
    conv1 = DepthwiseConv2D(kernel_size = (kernel,1), depth_multiplier=dm, activation='relu', padding=pad, kernel_initializer=init)(block_input)

    conv1 = Conv2D(filters*2, (3,1), activation='relu', padding=pad, kernel_initializer=init)(conv1)

    conv1 = Conv2D(filters * 1, (1, 1), activation=None, padding=pad, kernel_initializer=init)(conv1)

    skip_conv = Cropping2D(cropping=((2+adj, 2+adj), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])


    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def fe_block_s2(block_input, filters, init):

    pad = 'same'

    conv1 = Conv2D(filters*1, (3,1), strides=(2,1), activation = 'relu', padding = pad, kernel_initializer=init)(block_input)
    conv1 = Conv2D(filters*1, (3,1), activation = 'relu', padding = pad, kernel_initializer=init)(conv1)
    conv1 = Conv2D(filters * 1, (3, 1), activation=None, padding= pad, kernel_initializer=init)(conv1)

    # if np.shape(conv1)[-1] == np.shape(block_input)[-1]:
    #     res1 = Add()([conv1, block_input])
    # else:
    block_conv = Conv2D(filters, (1, 1), strides=(2, 1), activation=None, padding='same',
                        kernel_initializer=init)(block_input)
    # kernel_initializer=initializers.Constant(value=1))(block_input)
    res1 = Add()([conv1, block_conv])

    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def fe_conv_block(block_input, filters, init):
    pad = 'valid'
    conv1 = Conv2D(filters*1, (3,1), activation = 'relu', padding =pad, kernel_initializer=init)(block_input)
    conv1 = Conv2D(filters*1, (3,1), activation = 'relu', padding =pad, kernel_initializer=init)(conv1)
    conv1 = Conv2D(filters * 1, (3, 1), activation='relu', padding=pad, kernel_initializer=init)(conv1)
    conv1 = BatchNormalization()(conv1)

    return conv1

def fe_conv_block_valid(block_input, filters, init):
    conv1 = Conv2D(filters*1, (3,1), activation = 'relu', padding='valid', kernel_initializer=init)(block_input)
    conv1 = Conv2D(filters*1, (3,1), activation = 'relu', padding='valid', kernel_initializer=init)(conv1)
    conv1 = Conv2D(filters * 1, (3, 1), activation='relu', padding='valid', kernel_initializer=init)(conv1)
    conv1 = BatchNormalization()(conv1)

    return conv1

