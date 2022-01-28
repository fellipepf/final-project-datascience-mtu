from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Conv2D
import numpy as np
import pandas as pd
import tensorflow as tf


def example_1():
    # 1 dimension with 1 feature

    data = np.array([9,7,2,4,8,7,3,1,5,9,8,4])
    data = data.reshape(1, 12, 1)

    # define the kernel
    weights = np.array([1,3,6])
    weights = weights.reshape(3, 1, 1)
    bias = np.zeros((1,))  # 1 channel

    model = Sequential()
    model.add(Conv1D(1, 3, input_shape=(12, 1)))
    # conv1d = Conv1D(input, kernel, strides=1, padding='SAME')
    # #output_data = conv1d.convolution_op(input_data, kernel)

    # store the weights in the model
    model.set_weights([weights, bias])

    print(model.get_weights())
    yhat = model.predict(data)
    print(yhat)


def example_1_filters():
    # 1 dimension with 1 feature

    data = np.array([9,7,2,4,8,7,3,1,5,9,8,4])
    data = data.reshape(1, 12, 1)

    # define the kernel
    #filter=1
    #weights = np.array([1,3,6])
    #weights = weights.reshape(3, 1, 1)
    # filter=3
    weights = np.array([ [[1,3,6]], [[1,3,6]],[[1,3,6]] ])
    weights = weights.reshape(3, 1, 3)
    bias = np.zeros((1,))  # 1 channel
    bias = np.array([0.0 , 0.0, 0.0 ]) # 1 channel
    print(bias)

    model = Sequential()
    model.add(Conv1D(filters=3, kernel_size=3, input_shape=(12, 1)))
    # conv1d = Conv1D(input, kernel, strides=1, padding='SAME')
    # #output_data = conv1d.convolution_op(input_data, kernel)

    # store the weights in the model
    model.set_weights([weights, bias])

    print(model.get_weights())
    yhat = model.predict(data)
    print(yhat)

def example_2():
    # define input data
    data = [
        [10, 20, 30, 41, 51, 62, 73, 80, 66, 44, 86, 12],
        [0,   0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2],
        [1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    ]

    data = np.array(data).T
    df = pd.DataFrame(data, columns=['activity', 'category', 'time'])
   # df = df[[  'time','category','activity']]
    data = df.values
    d = []
    for columns in data:
        d.append(columns.reshape( 3, 1))

    print(d)
    #data = d

    data = np.array(data)
    #reshape(-1, window_size, N_FEATURES)
    data = data.reshape(-1, 12, 3)
    print(data.shape)
    print(data)

    model = Sequential()
    model.add(Conv1D(1, kernel_size=3,strides=1, input_shape=( 12, 3))) #(sample_number=Nonr,sample_size,channel_number)

    kernel_values = [
        [2,1,0],
        [2,1,0],
        [2,1,0]
    ]
    #kernel = [np.arrat(kernel_values).reshape( 3, 3,1), asarray([0.0])]
    bias = np.zeros((1,))  # 1 channel
    kernel = [np.array(kernel_values).reshape(3, 3, 1), bias]
    print(kernel)
    model.set_weights(kernel)
    print(model.get_weights())

    # apply filter to input data
    yhat = model.predict(data)
    print(yhat)
    #for r in range(yhat.shape[1]):
        # print each column in the row
    #    print([yhat[0, r, c, 0] for c in range(yhat.shape[2])])



if __name__ == '__main__':
    #example_1()
    example_1_filters()
    #example_2()

