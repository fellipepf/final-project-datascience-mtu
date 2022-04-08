''' ADaly's changes 2nd March 2022
Changed time window to do it by person - a little time inefficient - could save each person and reload
 26th March 2022'''
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
from keras.layers.convolutional import MaxPooling2D
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneGroupOut, LeavePGroupsOut
from sklearn.model_selection import LeaveOneOut
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

#my functions
import log_configuration
import my_metrics
from psykose import eda_baseline_date_range
from psykose_dataset import LoadDataset
from psykose_dataset import PreProcessing
from CNN_1d_AD1_1 import cnn_1d
LOGGER = log_configuration.logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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
    LOGGER.info("Creating time window, with person attribute...")

    window_size = 380
    N_FEATURES = 2
    step = 10

    segments = []
    labels = []
    classes = [c.name for c in df_dataset['class']]

    for i in range(0, df_dataset.shape[0] - window_size, step):
        x_activity = df_dataset['activity'].values[i: i + window_size]
        # x_person = df_dataset['user'].values[i: i + window_size]

        y_label = stats.mode(classes[i: i + window_size])[0][0]
        segments.append([x_activity])
        # segments.append([x_activity, x_person])
        # segments = np.vstack()
        # segments.append([x_activity])
        labels.append(y_label)


    # reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, window_size, N_FEATURES)
    # reshaped_segments = np.asarray(segments, dtype=np.float32).swapaxes(1,2)
    reshaped_segments = np.asarray(segments).swapaxes(1,2)
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    reshaped_segments.shape

    return reshaped_segments, labels


def split_dataset(x_values, y_values):
    LOGGER.info("Splitting dataset...")

    RANDOM_SEED = 24
    #x = dstack(x_values)
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=RANDOM_SEED)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def model_predict_k_fold(X_TRAIN, Y_TRAIN, n_splits=10, shuffle=True, random_state=2018):
    y_predicted_all = []
    y_true_all = []
    fold_no = 0

    k_fold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    start_time = datetime.now()

    # for train_index, fold_index in k_fold.split(np.zeros(len(X_TRAIN)), Y_TRAIN.ravel()):
    for train_index, fold_index in k_fold.split(np.zeros(len(Y_TRAIN)), Y_TRAIN[:, 0]):
        x_fold_train, x_fold_test = X_TRAIN[train_index, :], X_TRAIN[fold_index, :]
        y_fold_train, y_fold_test = Y_TRAIN[train_index, :], Y_TRAIN[fold_index, :]
        fold_no += 1
        print(f"Processing fold #{fold_no}")

        history_fit, test_accuracy, model, batch_size = run_cnn_model_1d(x_fold_train, x_fold_test, y_fold_train, y_fold_test)
        if fold_no == 1:
            print(model.summary())
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        plot_name_here = 'plot_fold_' + str(fold_no)
        result_plot(history_fit, plot_name=plot_name_here)

        y_predicted = model.predict(x_fold_test)
        y_predicted_arg = np.argmax(y_predicted, axis=1)
        y_test_arg = np.argmax(y_fold_test, axis=1)

        print(f"accuracy: {test_accuracy}" )

        modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
        metric_mattews_coef = modelMetrics.matthews_corrcoef()
        f1_score = modelMetrics.f1_score()
        accuracy = modelMetrics.accuracy()

        print(f" mcc: {metric_mattews_coef} - f1: {f1_score} - acc: {accuracy}")

        y_predicted_all.extend(list(y_predicted))
        y_true_all.extend(list(y_fold_test))

    return np.array(y_predicted_all), np.array(y_true_all), time_elapsed, history_fit, test_accuracy


def model_predict_lpgo(df_dataset, shuffle=True, random_state=2018):
    # metrics initialise
    mattews_coef = []
    f1_scores_full = []
    acc = []
    persons_score = []

    fold_no = 0
    groups = np.unique(df_dataset['user'])
    persons = np.unique(groups)

    start_time = datetime.now()

    # iterate over the lOO test person - and then split 43:10 training:validation person
    for person in persons:

        df_test = df_dataset[df_dataset['user'] == person]
        x_test, y_test = create_time_window(df_test)
        possible_categories = ['CONTROL','PATIENT']
        y_test = pd.Series(y_test)
        y_test = y_test.astype(pd.CategoricalDtype(categories=possible_categories))
        y_test = np.asarray(pd.get_dummies(y_test), dtype=np.float32)

        groups = groups[groups != person]
        persons_2 = persons[persons != person]
        groups_2 = np.random.choice(persons_2, 43, replace=False)
        val_persons = persons_2[persons_2 != groups_2]
        x_train = np.empty([1,380,1])
        y_train = []
        x_val = np.empty([1,380,1])
        y_val = []
        val_persons = []
        fold_no +=1

        for p in persons_2:
            if p in groups_2:
                df_train = df_dataset[df_dataset['user'] == p]
                x_train2, y_train2 = create_time_window(df_train)

                x_train = np.concatenate((x_train, x_train2), axis = 0)
                y_train = np.concatenate((y_train, y_train2), axis = 0)

            else:
                df_val = df_dataset[df_dataset['user'] == p]
                x_val2, y_val2 = create_time_window(df_val)

                x_val = np.concatenate((x_val, x_val2), axis = 0)
                y_val = np.concatenate((y_val, y_val2), axis = 0)
                val_persons.append(p)

        x_train = x_train[1:]         # remove empty concatenation
        y_train = pd.Series(y_train)
        y_train = y_train.astype(pd.CategoricalDtype(categories=possible_categories))
        y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
        x_val = x_val[1:]
        y_val = pd.Series(y_val)
        y_val = y_val.astype(pd.CategoricalDtype(categories=possible_categories))
        y_val = np.asarray(pd.get_dummies(y_val), dtype=np.float32)

        print(f"Processing test person #{person}")
        print(f"Processing validation persons #{val_persons}")
        print(f"Processing train persons #{groups_2}")
        history_fit, val_accuracy, model, batch_size = run_cnn_model_1d(x_train, x_val, y_train, y_val, person)
        if fold_no == 1:
            print(model.summary())
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        plot_name_here = 'plot_person_v2' + str(person)
        result_plot(history_fit, plot_name=plot_name_here)

        y_predicted = model.predict(x_test)
        y_predicted_arg = np.argmax(y_predicted, axis=1)
        y_test_arg = np.argmax(y_test, axis=1)

        print(f"val accuracy: {val_accuracy}" )

        modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
        # metric_mattews_coef = modelMetrics.matthews_corrcoef()
        # f1_score = modelMetrics.f1_score()
        accuracy = modelMetrics.accuracy()

        print(f" test metrics acc: {accuracy}")

        persons_score.append(person)
        # mattews_coef.append(metric_mattews_coef)
        # f1_scores_full.append(f1_score)
        acc.append(accuracy)

    print(f" full test acc metrics with person list:  mean_acc: {np.mean(acc)} persons {persons_score}  acc: {acc}")

    return time_elapsed


def run_cnn_model_1d(X_train, X_test, y_train, y_test, person):
    LOGGER.info("CNN model running...")

    verbose = 2
    epochs = 2
    batch_size = 32
    kernel = 3

    #sample =  X_train.shape[0]
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs =  y_train.shape[1]

    model = cnn_1d( n_timesteps, n_features, kernel=kernel)

    validation_data = (X_test, y_test)
    checkpoint_file_name = './Checkpoints/best_model' + str(person) + str('.h5')

    history_fit = model.fit(X_train, y_train, validation_data =validation_data, epochs=epochs, batch_size=batch_size,
                            callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                                         ModelCheckpoint(filepath=checkpoint_file_name, monitor='val_loss', save_best_only=True, verbose=1)],
                                         verbose=verbose)

    # evaluate model using best weights on val_loss
    saved_model = load_model(checkpoint_file_name)

    _, test_accuracy = saved_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    _, train_accuracy= saved_model.evaluate(X_train, y_train, batch_size=batch_size, verbose=verbose)
    print('Val: %.3f, Train: %.3f' % (test_accuracy, train_accuracy))
    return history_fit, test_accuracy, model, batch_size


def result_plot(history, plot_name='test'):
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
    # pyplot.show()
    plot_str = './Plots/' + str(plot_name) + '.png'
    plt.savefig(plot_str)
    plt.clf()

# def create_confusion_matrix(y_true, y_preds, classifier_name=None):
#     cm = confusion_matrix(y_true, y_preds, normalize='all')
#     cmd = ConfusionMatrixDisplay(cm, display_labels=['healthy', 'patient'])
#     cmd = cmd.plot(cmap=plt.cm.Blues, values_format='g')
#     cmd.ax_.set_title(f'Confusion Matrix - {classifier_name} ')
#     cmd.plot()
#     #cmd.ax_.set(xlabel='Predicted', ylabel='True')
#
#     plt.show()


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

    use_holdout_split =  False
    use_kfold_cross_validation = False #True
    use_leave_one_out = True # False

    check_options(use_kfold_cross_validation, use_leave_one_out, use_holdout_split)

    # x_values, y_values = create_time_window(df_dataset)


    if use_kfold_cross_validation:
        LOGGER.info("10 fold cross validation...")
        y_predicted, y_test, time_elapsed, history_fit, test_accuracy = model_predict_k_fold(df_dataset, n_splits=10, shuffle=True, random_state=2018)

        #df_result_metrics, df_classification_report_loo, df_feature_importance = decision_tree(df_result_metrics, df_classification_report_loo, df_feature_importance)

        # save_dataframe(df_result_metrics, result_filename)
        # save_dataframe(df_classification_report_loo, result_file_class_report_loo)


        result_plot(history_fit, plot_name='full_10_fold')

        # y_predicted = model.predict(X_test)
        y_predicted_arg = np.argmax(y_predicted, axis=1)
        y_test_arg = np.argmax(y_test, axis=1)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        print("10 fold cross validation full results...")
        print(f"accuracy: {test_accuracy}" )

        modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
        metric_mattews_coef = modelMetrics.matthews_corrcoef()
        f1_score = modelMetrics.f1_score()
        accuracy = modelMetrics.accuracy()

        print(f" mcc: {metric_mattews_coef} - f1: {f1_score} - acc: {accuracy}")


    if use_leave_one_out:
        LOGGER.info("Leave One Patient Out ...")

        # X_train, X_test, y_train, y_test = split_dataset(x_values, y_values)
        start_time = datetime.now()

        #evaluate_ltsm_model(X_train, X_test, y_train, y_test)
        y_predicted, y_test, time_elapsed, history_fit, test_accuracy = model_predict_lpgo(df_dataset, shuffle=True, random_state=2018)


        # history_fit, test_accuracy, model = run_cnn_model_2d(X_train, X_test, y_train, y_test)

        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    if use_kfold_cross_validation:
        LOGGER.info("10 fold cross validation...")
    y_predicted, y_test, time_elapsed, history_fit, test_accuracy = model_predict_k_fold(df_dataset, n_splits=10, shuffle=True, random_state=2018)

    #df_result_metrics, df_classification_report_loo, df_feature_importance = decision_tree(df_result_metrics, df_classification_report_loo, df_feature_importance)

    # save_dataframe(df_result_metrics, result_filename)
    # save_dataframe(df_classification_report_loo, result_file_class_report_loo)


    result_plot(history_fit, plot_name='full_10_fold')

    # y_predicted = model.predict(X_test)
    y_predicted_arg = np.argmax(y_predicted, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print("10 fold cross validation full results...")
    print(f"accuracy: {test_accuracy}" )

    modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
    metric_mattews_coef = modelMetrics.matthews_corrcoef()
    f1_score = modelMetrics.f1_score()
    accuracy = modelMetrics.accuracy()

    print(f" mcc: {metric_mattews_coef} - f1: {f1_score} - acc: {accuracy}")
