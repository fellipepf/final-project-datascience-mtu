
import os
import sys, inspect
import numpy as np
import pandas as pd

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot

from scipy import stats

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from datetime import datetime

from tensorflow import keras

from psykose import LoadDataset, PreProcessing, eda_baseline_date_range


#for each user
def process_dataset(dataset):
    df_stats = pd.DataFrame()
    x_values = list()
    y_values = list()

    number_days_analyse = 4
    remove_first_day = True
    for key in set(dataset.keys()):
        value = dataset[key]

        df = value['timeserie']
        user_class = value['target']

        df["datetime"] = pd.to_datetime(df["timestamp"])
        #group_day = df.groupby(df["datetime"].dt.day)['activity']

        group_day = df.groupby(pd.Grouper(key='datetime', freq='D'))
        list_days = list(group_day)

        #remove first day and
        # slice n number of elements defined
        group_n_days = list_days[1:number_days_analyse +1]



        #get the second element of the tuple that is the timeseries dataframe
        list_tm = [tuple[1] for tuple in group_n_days]

        #concat list of dataframes into one dataframe
        df_tm = pd.concat(list_tm)

        #
        values = df_tm['activity'].values
        #values = values.reshape(-1, 480)
        x_values.append(values)
        y_values.append(user_class.value)



    return x_values, y_values

def create_time_window(df_dataset):
    N_TIME_STEPS = 50  # 50 records in each sequence
    N_FEATURES = 1  # x, y, z
    step = 10  # window overlap = 50 -10 = 40  (80% overlap)
    N_CLASSES = 2  # class labels



    segments = []
    labels = []
    classes = [c.name for c in df_dataset['class']]
    for i in range(0, df_dataset.shape[0] - N_TIME_STEPS, step):
        xs = df_dataset['activity'].values[i: i + 50]

        label = stats.mode(classes[i: i + 50])[0][0]
        segments.append([xs])
        labels.append(label)

    # reshape the segments which is (list of arrays) to one list
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    reshaped_segments.shape
    print(labels)
    print(labels.shape)

    return reshaped_segments, labels


def split_dataset(x_values, y_values):
    RANDOM_SEED = 24
    #x = dstack(x_values)
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=RANDOM_SEED)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def evaluate_ltsm_model(X_train, X_test, y_train, y_test):
    # LSTM model
    print("LSTM model running...")

    epochs = 50
    batch_size = 1024
    verbose = 1

    model = Sequential()
    # RNN layer
    model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Dropout layer
    model.add(Dropout(0.5))
    # Dense layer with ReLu
    model.add(Dense(units=64, activation='relu'))
    # Softmax layer
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # Compile model
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    model.summary()


    # Training
    history = model.fit(X_train, y_train, epochs = epochs,
                    validation_split = 0.25, batch_size = batch_size, verbose = verbose)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)


def evaluate_cnn_model(X_train, X_test, y_train, y_test):
    print("CNN model running...")
    verbose, epochs, batch_size = 1, 10, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


if __name__ == '__main__':
    dict_dataset = LoadDataset().get_dataset_joined()
    preProcessing = PreProcessing(dict_dataset)
    df_dataset = preProcessing.df_dataset

    x_values, y_values = create_time_window(df_dataset)

    #x_values, y_values = process_dataset(control | patient )
    X_train, X_test, y_train, y_test = split_dataset(x_values, y_values)

    start_time = datetime.now()

    evaluate_ltsm_model(X_train, X_test, y_train, y_test)
    #evaluate_cnn_model(X_train, X_test, y_train, y_test)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    #range_info = eda_baseline_date_range(control)

    #print(range_info)

