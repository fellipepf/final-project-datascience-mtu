from numpy import asarray
import numpy as np
import pandas as pd
import mxnet as mx


#https://mxnet.apache.org/versions/1.1.0/api/python/gluon/nn.html
#https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108

def apply_conv(data, kernel, conv):
    """
    Args:
        data (NDArray): input data.
        kernel (NDArray): convolution's kernel parameters.
        conv (Block): convolutional layer.
    Returns:
        NDArray: output data (after applying convolution).
    """
    # add dimensions for batch and channels if necessary
    while data.ndim < len(conv.weight.shape):
        data = data.expand_dims(0)
    # add dimensions for channels and in_channels if necessary
    while kernel.ndim < len(conv.weight.shape):
        kernel = kernel.expand_dims(0)
    # check if transpose convolution
    if type(conv).__name__.endswith("Transpose"):
        in_channel_idx = 0
    else:
        in_channel_idx = 1
    # initialize and set weight
    conv._in_channels = kernel.shape[in_channel_idx]
    conv.initialize()
    conv.weight.set_data(kernel)
    return conv(data)

run_ex_1 = True
if run_ex_1:
    input_data = mx.nd.array((9,7,2,4,8,7,3,1,5,9,8,4))
    kernel = mx.nd.array((1,3,6))

    conv = mx.gluon.nn.Conv1D(channels=1, kernel_size=3,strides=1)
    output_data = apply_conv(input_data, kernel, conv)
    print(output_data)

run_ex_2 = False
if run_ex_2:
    input_data = mx.nd.array((
        (1, 10, 2, 1),  #feature1 = time series
        (0, 0, 1, 1)  #feature2  = category
                              ))

    kernel = mx.nd.array(((1, 2, 3),
                          (0, 1, 0)
                          ))

    conv = mx.gluon.nn.Conv1D(channels=1,  #its 2 chanels but 1D means slides along 1 dimension so use 1 channel
                              kernel_size=3,
                              strides=1,
                              padding=0)
    output_data = apply_conv(input_data, kernel, conv)
    print(output_data)