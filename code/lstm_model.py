
import os
import sys, inspect
import numpy as np
import pandas as pd

from datetime import datetime
from matplotlib import pyplot, pyplot as plt
from numpy import dstack
from scipy import stats

# cnn model
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#my functions
import log_configuration
import my_metrics
from psykose import LoadDataset, PreProcessing, eda_baseline_date_range

LOGGER = log_configuration.logger

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
    LOGGER.info("Creating time window...")

    window_size = 380
    N_FEATURES = 2
    step = 10

    segments = []
    labels = []
    classes = [c.name for c in df_dataset['class']]
    for i in range(0, df_dataset.shape[0] - window_size, step):
        x_activity = df_dataset['activity'].values[i: i + window_size]
        x_category = df_dataset['category'].values[i: i + window_size]

        y_label = stats.mode(classes[i: i + window_size])[0][0]

        segments.append([x_activity, x_category])
        labels.append(y_label)


    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, window_size, N_FEATURES)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    reshaped_segments.shape
    print(labels)
    print(labels.shape)

    return reshaped_segments, labels


def split_dataset(x_values, y_values):
    LOGGER.info("Splitting dataset...")

    RANDOM_SEED = 24
    #x = dstack(x_values)
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=RANDOM_SEED)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def evaluate_ltsm_model(X_train, X_test, y_train, y_test):
    # LSTM model
    print("LSTM model running...")

    epochs = 20
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


def evaluate_cnn_model_2d(X_train, X_test, y_train, y_test):
    logger = log_configuration.logger
    logger.info("CNN model running...")

    verbose = 1
    epochs = 30
    batch_size = 32

    sample =  X_train.shape[0]
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs =  y_train.shape[1]

    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=4, activation='relu', input_shape=(sample, n_timesteps, n_features)))
    model.add(Conv2D(filters=128, kernel_size=4, activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=4, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


def run_cnn_model_working(X_train, X_test, y_train, y_test):
    LOGGER.info("CNN model running...")

    verbose = 1
    epochs = 30
    batch_size = 32

    #sample =  X_train.shape[0]
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs =  y_train.shape[1]

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=( n_timesteps, n_features)))
    #model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # fit network
    validation_data = (X_test, y_test)
    history_fit = model.fit(X_train, y_train, validation_data =validation_data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return history_fit, test_accuracy, model


def result_plot(history):
    print(history.history.keys())
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

def create_confusion_matrix(y_true, y_preds, classifier_name=None):
    cm = confusion_matrix(y_true, y_preds, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['healthy', 'patient'])
    cmd = cmd.plot(cmap=plt.cm.Blues, values_format='g')
    cmd.ax_.set_title(f'Confusion Matrix - {classifier_name} ')
    cmd.plot()
    #cmd.ax_.set(xlabel='Predicted', ylabel='True')

    plt.show()

def check_options(*options):
    '''
    Verify if only one validation/split method is chosen
    If more than one the script stop running
    :param options:
    :return:
    '''

    try:
        assert sum(options) == 1
    except Exception as e:
        LOGGER.error("Only one option must be chosen")
        exit()

if __name__ == '__main__':
    LOGGER.info("Script started...")

    dict_dataset = LoadDataset().get_dataset_joined()
    preProcessing = PreProcessing(dict_dataset)
    df_dataset = preProcessing.df_dataset_category

    use_k_fold_cross_validation = False
    use_leave_one_out = False
    use_traditional_split = True

    check_options(use_k_fold_cross_validation, use_leave_one_out, use_traditional_split)

    x_values, y_values = create_time_window(df_dataset)

    #x_values, y_values = process_dataset(control | patient )
    X_train, X_test, y_train, y_test = split_dataset(x_values, y_values)

    start_time = datetime.now()

    #evaluate_ltsm_model(X_train, X_test, y_train, y_test)
    #evaluate_cnn_model(X_train, X_test, y_train, y_test)

    history_fit, test_accuracy, model = run_cnn_model_working(X_train, X_test, y_train, y_test)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    result_plot(history_fit)

    y_predicted = model.predict(X_test)
    y_predicted_arg = np.argmax(y_predicted, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)

    print(f"accuracy: {test_accuracy}" )



    modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
    metric_mattews_coef = modelMetrics.matthews_corrcoef()
    f1_score = modelMetrics.f1_score()
    accuracy = modelMetrics.accuracy()

    print(f" mcc: {metric_mattews_coef} - f1: {f1_score} - acc: {accuracy}")

    create_confusion_matrix(y_test_arg, y_predicted_arg, "CNN")



