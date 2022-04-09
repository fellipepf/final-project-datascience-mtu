import enum
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd

import pprint
from tabulate import tabulate
import dataframe_image as dfi

import matplotlib.pyplot as plt
import seaborn as sns

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier

#scikit-learn
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
from sklearn.base import BaseEstimator

# my functions
import log_configuration
import my_metrics
import psykose_machine_learning_plots as results_plot
import hyperparameter_tuning as tuning
import utils
from utils import get_name_from_value
from utils import Target


color = sns.color_palette()

sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py

LOGGER = log_configuration.logger

# provided baseline dataset
#PATH_TO_FILE = "../psykose/schizophrenia-features.csv"

# baseline dataset with new features
PATH_TO_FILE = "baseline_time_period.csv"
#PATH_TO_FILE = "baseline_time_period_6_periods.csv"
#PATH_TO_FILE = "baseline_time_period_4_periods.csv"

# baseline dataset with features defined on published paper
# reproduced in this research
#PATH_TO_FILE = "my_baseline.csv"

LOGGER.info(f"Dataset selected: {PATH_TO_FILE}")





def load_dataset():
    data = pd.read_csv(PATH_TO_FILE)
    return data


data = load_dataset()
dataX = data.copy().drop(["class", "class_str", "userid"], axis=1)
dataY = data["class"].copy()


scaler = pp.StandardScaler(copy=True)
dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])


class TrainTestSets:
    '''
    Dataclass for each training set and testing set (one person out)
    '''
    def __init__(self, x_train, x_test, y_train, y_test, id_person_out):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.id_person_out = id_person_out

    def calc_metrics(self, y_test_predicted):
        model_metrics = my_metrics.ModelMetrics(self.y_test, y_test_predicted)
        return model_metrics





def holdout():
    testset_size = 0.33
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        dataX,
        dataY,
        test_size=testset_size,
        random_state=2019,
        stratify=dataY   #keep the proportion of the classes for each subset
    )

    train_test_sets = TrainTestSets(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, "hold-out")

    return list(train_test_sets)

def get_userid_from_index(index):
    users_list = data["userid"].unique()
    return users_list[index]

def get_content_fom_person(data, index):
    userid = get_userid_from_index(index)
    content = data.loc[data['userid'].isin(userid)]

    dataX = content.copy().drop(["class", "class_str", "userid"], axis=1)
    dataY = content["class"].copy()

    return dataX, dataY

def scale_data(dataX):
    scaler = pp.StandardScaler(copy=True)
    dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])
    return dataX

def leave_one_patient_out():
    list_traning_test_set = list()
    data = load_dataset()


    users_list = data["userid"].unique()
    loo = LeaveOneOut()
    loo.get_n_splits(users_list)

    print(loo)
    for train_index, test_index in loo.split(users_list):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_content_x, train_content_y = get_content_fom_person(data, train_index)
        test_content_x, test_content_y = get_content_fom_person(data, test_index)

        person_out = get_userid_from_index(test_index)[0]
        train_test_sets = TrainTestSets(train_content_x, test_content_x, train_content_y, test_content_y, person_out)

        list_traning_test_set.append(train_test_sets)

    return list_traning_test_set


def create_training_test_sets(testset_size = 0.33, method = None ):
    train_test_sets_result = list()

    if method == "holdout":
        train_test_sets_result = holdout()

    if method == "lopo":
        train_test_sets_result = leave_one_patient_out()

    return train_test_sets_result




class ValidationMethod(enum.Enum):
    KFold = "K-Fold Cross-Validation"
    LOO = "Leave-One-Out"

class Classifier(enum.Enum):
    log_reg = "Logistic Regression"
    r_forest = "Random Forest"
    d_tree = "Decision Tree"
    xgb = "XGBoost"
    lgbm = "LightGBM"

class Metric(enum.Enum):
    f1_score = "F1-Score"




###
#Refactoring ML Models
###

class ModelStructure(BaseEstimator):

    def __init__(self, name, model, params ):

        self.name = name
        self.model = model
        self.params = params

        #self.model.set_params(**params)

    def fit(self, X, y=None, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]


#####

def get_parameters():


    _PARAMS_LOGREG = {
        "penalty": "l2",
        "C": 100,
        "class_weight": "balanced",
        "random_state": 2018,
        "solver": "saga",
        "n_jobs": 1
    }

    _PARAMS_RFC = {
        "n_estimators": 200,
        "max_features": "auto",
        "max_depth": 100,
        "min_samples_split": 8,
        "min_samples_leaf": 5,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": 10,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": 2018,
        "class_weight": "balanced"
    }

    _PARAMS_DTC = {
        'criterion': 'entropy',
        "max_features": 'sqrt',
        "max_depth": 10,
        "min_samples_split": 6,
        "min_samples_leaf": 4,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": 12,
        "random_state": 2018,
        "class_weight": "balanced"

    }

    _PARAMS_XGB = {
        "nthread": 16,
        "learning_rate": 0.3,
        "gamma": 0,
        "max_depth": 7,
        "verbosity": 0,
        "min_child_weight": 3,
        "max_delta_step": 0,
        "subsample": 0.5,
        "colsample_bytree": 0.7,
        "objective": "binary:logistic",
        "num_class": 1,
        "eval_metric": "logloss",
        "seed": 2018,
    }

    _PARAMS_LIGHTGB = {
        "task": "train",
        "num_class": 1,
        "boosting": "gbdt",
        "verbosity": -1,
        "objective": "binary", "metric": "binary_logloss", "metric_freq": 50, "is_training_metric": False,
        "max_depth": 4, "num_leaves": 31, "learning_rate": 0.1, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 0, "bagging_seed": 2018, "num_threads": 16
    }

    params_dic = {}
    params_dic['LR'] = _PARAMS_LOGREG
    params_dic['RF'] = _PARAMS_RFC
    params_dic['DT'] = _PARAMS_DTC
    params_dic['XB'] = _PARAMS_XGB
    params_dic['LG'] = _PARAMS_LIGHTGB

    return params_dic


def get_models():
    models = dict()

    params = get_parameters()

    #models['LR'] = (ModelStructure("Logistic Regression", LogisticRegression(**params.get('LR')), ""))
    #models['RF'] = (ModelStructure("Random Forest", RandomForestClassifier(), ""))
    #models['DT'] = (ModelStructure("Random Forest", DecisionTreeClassifier(), ""))
    #models['XB'] = (ModelStructure("XGBoost", xgb.XGBClassifier(**params.get('XB')), ""))
    #models['LG'] = (ModelStructure("LightGBM", lgb.LGBMClassifier(**params.get('LG')), ""))
    models['CB'] = (ModelStructure("CatBoost", CatBoostClassifier(), ""))

    return models

def check_options(*options):
    '''
    Verify if only one validation/split method is chosen
    If more than one the script stop running
    :param options:
    :return:
    '''
    logger = log_configuration.logger

    try:
        assert sum(options) == 1
    except Exception as e:
        logger.error("Only one option must be chosen")
        exit()


def save_dataframe(dataframe, name=None):
    if name:
        file_name = f"df_results/df_result_{name}.pkl"
    else:
        file_name = "df_results/df_result.pkl"

    dataframe.to_pickle(file_name)

def load_dataframe(name=None):
    if name:
        file_name = f"df_results/df_result_{name}.pkl"
    else:
        file_name = "df_results/df_result.pkl"

    output = pd.read_pickle(file_name)
    return output

if __name__ == '__main__':
    LOGGER.info("Script started...")



    run_hyper_tuning = False
    run_models = True
    read_result_df_saved = False
    check_options(run_hyper_tuning, run_models, read_result_df_saved)

    train_test_sets = create_training_test_sets(method="lopo")

    result_loo = dict()
    #result_loo['user'] = None
    dict_result_user = dict()

    #models = list()
    #models.append(ModelStructure("Logistic Regression", LogisticRegression(), ""))
    #models.append(ModelStructure("Random Forest", RandomForestClassifier(), ""))
    #models.append(ModelStructure("XGBoost", xgb.XGBClassifier(), ""))
    #models.append(ModelStructure("LightGBM", lgb.LGBMClassifier(), ""))

    models = get_models()

    # iterate models
    for key, model in models.items():
        LOGGER.info(f"{model.name}")

        # iterate each person on test set
        list_acc = list()
        for train_test_set in train_test_sets:
            #doubt: use the same instance of the model for each observation ?
            model = models.get(key)

            model.fit(train_test_set.x_train, train_test_set.y_train)
            y_test_preds = model.predict_proba(train_test_set.x_test)

            acc = train_test_set.calc_metrics(y_test_preds).accuracy()
            LOGGER.info(f"{model.name} - {train_test_set.id_person_out} - acc: {acc}")
            #dict_result_user[train_test_set.id_person_out] = acc
            list_acc.append(acc)

        result_loo[model.name] = np.mean(list_acc)

    print(result_loo)
