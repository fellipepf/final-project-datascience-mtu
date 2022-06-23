''' ADaly's changes 2nd March 2022
Changed time window to do it by person - a little time inefficient - could save each person and reload
 26th March 2022
 window 1440 step 360
 81% best so far
 6_4 mvoing average
 _6t - six time periods using psykose_dataset_6.py '''
import os
import sys, inspect
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot, pyplot as plt
from numpy import dstack
from scipy import stats

from tabulate import tabulate

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
from psykose_eda import eda_baseline_date_range
from psykose_dataset_6 import LoadDataset
from psykose_dataset_6 import PreProcessing
from CNN_2d_AD1_1 import cnn_2D
LOGGER = log_configuration.logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
label = '3_hour_noval_6t_'
window_size = 7


def movingaverage(data, window):
    '''

    :param data: the vector which the MAF will be applied to
    :param window: the size of the moving average window
    :return: data after the MAF has been applied
    '''
    data = data
    window = np.ones(int(window)) / float(window_size)
    return np.convolve(data, window, "same")



# for each user
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
    # LOGGER.info("Creating time window, with person attribute...")

    window_size = 1440
    N_FEATURES = 2
    step = 180

    segments = []
    labels = []
    classes = [c.name for c in df_dataset['class']]

    for i in range(0, df_dataset.shape[0] - window_size, step):
        x_activity = df_dataset['activity'].values[i: i + window_size]
        x_time_cat = df_dataset['category'].values[i: i + window_size]
        # x_person = df_dataset['user'].values[i: i + window_size]

        y_label = stats.mode(classes[i: i + window_size])[0][0]
        segments.append([x_activity, x_time_cat])
        # segments = np.vstack()
        # segments.append([x_activity])
        labels.append(y_label)


    # reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, window_size, N_FEATURES)
    # reshaped_segments = np.asarray(segments, dtype=np.float32).swapaxes(1,2)
    reshaped_segments = np.asarray(segments).swapaxes(1,2)
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    reshaped_segments.shape

    return reshaped_segments, labels

## begin - metrics

class PredictedValues():
    def __init__(self, y_true, y_predicted, model_name, id_person_out ):
        self.y_true = y_true
        self.y_predicted = y_predicted
        self.model_name = model_name
        self.id_person_out = id_person_out

    def calc_metrics(self):
        model_metrics = my_metrics.ModelMetrics(self.y_true, self.y_predicted)
        return model_metrics

    def is_patient(self):
        return self.__get_class_person(self.id_person_out) == "patient"

    def __get_class_person(self, id):
        result = id.split("_")[0]
        return result



def metric_values(person_list):
    result_df_lopo_per_person = pd.DataFrame()

    for person_values in person_list:

        acc = person_values.calc_metrics().accuracy()

        precision_micro = person_values.calc_metrics().precision("micro")
        precision_macro = person_values.calc_metrics().precision("macro")
        precision = person_values.calc_metrics().precision()

        recall = person_values.calc_metrics().recall(average='micro')
        f1_score = person_values.calc_metrics().f1_score(average='micro')
        mcc = person_values.calc_metrics().matthews_corrcoef()

        if person_values.is_patient():
            person_class = "Patient"
        else:
            person_class = "Control"

        result_metrics_dict = dict()
        result_metrics_dict["model"] = "CNN"
        result_metrics_dict["class"] = person_class

        result_metrics_dict["accuracy"] = acc
        result_metrics_dict["precision"] = precision_micro
        result_metrics_dict["f1-score"] = f1_score
        result_metrics_dict["recall"] = recall
        result_metrics_dict["mcc"] = mcc

        result_df_lopo_per_person = result_df_lopo_per_person.append(result_metrics_dict, ignore_index=True)


    return result_df_lopo_per_person

def average_metric_values(df_class_person):

    mean_acc = df_class_person[["accuracy"]].mean()
    mean_precision = df_class_person[["precision"]].mean()
    mean_recall = df_class_person[["recall"]].mean()
    mean_f1_score = df_class_person[["f1-score"]].mean()
    mcc_mean = df_class_person[['mcc']].mean()

    #columns output
    result = dict()
    result["model"] = df_class_person["model"].unique()
    result["class"] = df_class_person["class"].unique()
    result['accuracy'] = mean_acc[0]
    result['precision'] = mean_precision[0]
    result['recall'] = mean_recall[0]
    result['f1-score'] = mean_f1_score[0]

    return result

def metric_calculation(predicted_values_dict):
    result_df = pd.DataFrame()

    #filter the list and return values for each class
    control_person = [person_values for person_values in predicted_values_dict
                      if person_values.is_patient() == False]

    patient_person = [person_values for person_values in predicted_values_dict
                      if person_values.is_patient() == True]

    #control
    control_metrics = metric_values(control_person)
    result_metics_control = average_metric_values(control_metrics)
    result_df = result_df.append(result_metics_control, ignore_index=True)

    #patient
    patient_metrics = metric_values(patient_person)
    result_metics_patient = average_metric_values(patient_metrics)
    result_df = result_df.append(result_metics_patient, ignore_index=True)

    #weighted
    weighted_metrics = metric_values(predicted_values_dict)
    result_metics_weighted = average_metric_values(weighted_metrics)
    result_df = result_df.append(result_metics_weighted, ignore_index=True)

    print(tabulate(result_df, headers='keys', tablefmt='psql'))

## end - metrics

def model_predict_lpgo(df_dataset, shuffle=True, random_state=2018):
    # metrics initialise
    mattews_coef = []
    f1_scores_full = []
    acc = []
    acc_nmov = []
    persons_score = []

    fold_no = 0
    groups = np.unique(df_dataset['user'])
    persons = np.unique(groups)

    #first 6 persons to make it fast to run and debug
    '''
    import random
    random.shuffle(persons)
    persons = persons[:6]
    '''

    list_predicted_values = list()

    start_time = datetime.now()

    # iterate over the lOO test person - and then split 43:10 training:validation person
    for person in persons:

        df_test = df_dataset[df_dataset['user'] == person]
        x_test, y_test = create_time_window(df_test)
        x_test = np.expand_dims(x_test,-1)
        possible_categories = ['CONTROL','PATIENT']
        y_test = pd.Series(y_test)
        y_test = y_test.astype(pd.CategoricalDtype(categories=possible_categories))
        y_test = np.asarray(pd.get_dummies(y_test), dtype=np.float32)


        dims = x_test.shape[2]
        groups = groups[groups != person]
        persons_2 = persons[persons != person]
        groups_2 = np.random.choice(persons_2, 53, replace=False)
        val_persons = persons_2[persons_2 != groups_2]
        x_train = np.empty([1,1440,dims])
        y_train = []
        x_val = np.empty([1,1440,dims])
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

        x_train = np.expand_dims(x_train[1:],-1)
        x_train = np.asarray(x_train).astype(np.float32)
        y_train = pd.Series(y_train)
        y_train = y_train.astype(pd.CategoricalDtype(categories=possible_categories))
        y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
        x_val = np.expand_dims(x_val[1:], -1)
        y_val = pd.Series(y_val)
        y_val = y_val.astype(pd.CategoricalDtype(categories=possible_categories))
        y_val = np.asarray(pd.get_dummies(y_val), dtype=np.float32)

        print(f"Processing test person #{person}")
        print(f"Processing validation persons #{val_persons}")
        print(f"Processing train persons #{groups_2}")
        history_fit, val_accuracy, model, batch_size = run_cnn_model_2d(x_train, x_val, y_train, y_val, person)
        if fold_no == 1:
            print(model.summary())
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        plot_name_here = 'plot_person_v6_1_cnn2d' +str(label) + str(person)
        result_plot(history_fit, plot_name=plot_name_here)

        x_test = np.asarray(x_test).astype(np.float32)
        y_predicted = model.predict(x_test)
        y_predicted_arg = movingaverage(model.predict(x_test)[:,1], window_size)
        y_predicted_arg2 = np.argmax(y_predicted, axis=1)
        y_test_arg = np.argmax(y_test, axis=1)

        _, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

        print(f"val accuracy: {val_accuracy}" )

        modelMetrics = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg)
        modelMetrics2 = my_metrics.ModelMetrics(y_test_arg, y_predicted_arg2)
        #metric_mattews_coef = modelMetrics.matthews_corrcoef()
        f1_score = modelMetrics.f1_score(average='weighted')
        accuracy = modelMetrics.accuracy()
        accuracy_nmov = modelMetrics2.accuracy()

        #print(f" test metrics mcc: {metric_mattews_coef}")
        print(f" test metrics f1_score: {f1_score}")
        print(f" test metrics acc_mov: {accuracy}")
        print(f" test metrics acc_no_mov: {accuracy_nmov}")
        print(f" test metrics acc_eval: {test_accuracy}")

        persons_score.append(person)
        # mattews_coef.append(metric_mattews_coef)
        # f1_scores_full.append(f1_score)
        acc.append(accuracy)
        acc_nmov.append(accuracy_nmov)

        #collecting values to calculate the metrics
        predicted_values_obj = PredictedValues(y_test_arg, y_predicted_arg, model.name, person)
        list_predicted_values.append(predicted_values_obj)
    metric_calculation(list_predicted_values)

    print(f" full test acc metrics with person list:  mean_acc: {np.mean(acc)} mean_no_mov_avg_acc: {np.mean(acc_nmov)} persons {persons_score}  acc: {acc}")


def run_cnn_model_2d(X_train, X_test, y_train, y_test, person):
    LOGGER.info("CNN model running...")

    verbose = 2
    epochs = 2
    batch_size = 32
    kernel = (3,2)

    #sample =  X_train.shape[0]
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs =  y_train.shape[1]

    model = cnn_2D( n_timesteps, n_features, kernel=kernel)

    validation_data = (X_train, y_train) # no val here
    checkpoint_file_name = './Checkpoints/best_model_CNN2D_val2' + str(person) + str('.h5')
    history_fit = model.fit(X_train, y_train, validation_data =validation_data, epochs=epochs, batch_size=batch_size,
                            callbacks = [EarlyStopping(monitor='val_accuracy', patience=4), # was 7 for best
                                         ModelCheckpoint(filepath=checkpoint_file_name, monitor='val_accuracy', save_best_only=True, verbose=1)],
                            verbose=verbose)

    # evaluate model using best weights on val_loss
    saved_model = load_model(checkpoint_file_name)

    _, test_accuracy = saved_model.evaluate(X_train, y_train, batch_size=batch_size, verbose=verbose)
    # _, test_accuracy = model(X_test, training = False, verbose=verbose)
    _, train_accuracy= saved_model.evaluate(X_train, y_train, batch_size=batch_size, verbose=verbose)
    print('Val: %.3f, Train: %.3f' % (test_accuracy, train_accuracy))

    return history_fit, test_accuracy, saved_model, batch_size


def folder_to_save_plots():
    '''
    Fellipe

    :return:
    '''

    folder = './Plots/'
    if not os.path.exists(folder):
        os.mkdir(folder)
        return folder
    else:
        return folder



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
    folder = folder_to_save_plots()
    plot_str = folder + str(plot_name) + '.png'
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


    if use_leave_one_out:
        LOGGER.info("Leave One Patient Out ...")

        # X_train, X_test, y_train, y_test = split_dataset(x_values, y_values)
        start_time = datetime.now()

        #evaluate_ltsm_model(X_train, X_test, y_train, y_test)
        model_predict_lpgo(df_dataset, shuffle=True, random_state=2018)


        # history_fit, test_accuracy, model = run_cnn_model_2d(X_train, X_test, y_train, y_test)
