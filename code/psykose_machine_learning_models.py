import enum
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from tabulate import tabulate
import dataframe_image as dfi

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut

#from matplotlib import pyplot
from xgboost import plot_importance
import log_configuration
import my_metrics
import psykose_machine_learning_plots as results_plot
import hyperparameter_tuning as tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime

color = sns.color_palette()

sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py
import utils
from utils import get_name_from_value
from utils import Target

LOGGER = log_configuration.logger


# provided baseline dataset
#PATH_TO_FILE = "../psykose/schizophrenia-features.csv"


# baseline dataset with new features
#PATH_TO_FILE = "baseline_time_period.csv"

# baseline dataset with features defined on published paper
# reproduced in this research
PATH_TO_FILE = "my_baseline.csv"

LOGGER.info(f"Dataset selected: {PATH_TO_FILE}")


_PARAMS_LORGREG = {
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



def load_dataset():
    data = pd.read_csv(PATH_TO_FILE)
    return data

data = load_dataset()
dataX = data.copy().drop(["class", "class_str", "userid"], axis=1)
dataY = data["class"].copy()


scaler = pp.StandardScaler(copy=True)
dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])


testset_size = 0.33

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    dataX,
    dataY,
    test_size=testset_size,
    random_state=2019,
    stratify=dataY
)


def plot_prc_curve(y_preds, y_trues, title=None):
    precision, recall, _ = metrics.precision_recall_curve(
        y_trues,
        y_preds
    )

    average_precision = metrics.average_precision_score(
        y_trues,
        y_preds
    )
    #clear plots from last use
    plt.close()

    print("Average Precision = %.2f" % average_precision)
    plt.step(recall, precision, color="k", alpha=0.7, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.3, color="k")

    if title is None:
        title = "PRC: Average Precision = %.2f" % average_precision

    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

    return average_precision


def plot_roc_curve(y_preds, y_trues, title=None):
    fpr, tpr, _ = metrics.roc_curve(y_trues, y_preds)
    auc_roc = metrics.auc(fpr, tpr)

    print("AUCROC = %.2f" % auc_roc)

    if title is None:
        title = "AUCROC = %.2f" % auc_roc

    #clear plots from last use
    plt.close()

    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, color="r", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
    plt.show()

    return auc_roc

def plot_traning_curves(rfc_y_preds, rfc_y_trues):
    plot_prc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Validation PRC")
    plot_roc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Validation ROC")


def plot_testing_curves(rfc_test_preds, Y_TEST):
    plot_prc_curve(rfc_test_preds, Y_TEST, "RFC Testing PRC")
    plot_roc_curve(rfc_test_preds, Y_TEST, "RFC Testing ROC")

def model_predict_k_fold(train_func, pred_func, model=None, n_splits=10, shuffle=True, random_state=2018):
    y_preds = []
    y_trues = []

    k_fold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    start_time = datetime.now()

    for train_index, fold_index in k_fold.split(np.zeros(len(X_TRAIN)), Y_TRAIN.ravel()):
        x_fold_train, x_fold_test = X_TRAIN.iloc[train_index, :], X_TRAIN.iloc[fold_index, :]
        y_fold_train, y_fold_test = Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[fold_index]

        model = train_func(model, x_fold_train, y_fold_train, x_fold_test, y_fold_test)
        y_pred = pred_func(model, x_fold_test)

        y_preds.extend(list(y_pred))
        y_trues.extend(list(y_fold_test))

    time_elapsed = datetime.now() - start_time

    return model, np.array(y_preds), np.array(y_trues), time_elapsed


def leave_one_out(train_func, pred_func, model=None):
    y_preds = []
    y_trues = []

    loo = LeaveOneOut()

    start_time = datetime.now()

    for train_index, fold_index in loo.split(np.zeros(len(X_TRAIN)), Y_TRAIN.ravel()):
        x_fold_train, x_fold_test = X_TRAIN.iloc[train_index, :], X_TRAIN.iloc[fold_index, :]
        y_fold_train, y_fold_test = Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[fold_index]

        model = train_func(model, x_fold_train, y_fold_train, x_fold_test, y_fold_test)
        y_pred = pred_func(model, x_fold_test)

        y_preds.extend(list(y_pred))
        y_trues.extend(list(y_fold_test))

    time_elapsed = datetime.now() - start_time

    return model, np.array(y_preds), np.array(y_trues), time_elapsed

def create_confusion_matrix(y_true, y_preds, classifier_name=None, method_short_name=None):
    cm = confusion_matrix(y_true, y_preds) #, normalize='all'
    cmd = ConfusionMatrixDisplay(cm, display_labels=['healthy', 'patient'])
    cmd = cmd.plot(cmap=plt.cm.Blues, values_format='g')
    cmd.ax_.set_title(f'Confusion Matrix - {classifier_name} - {method_short_name}')
    cmd.plot()
    #cmd.ax_.set(xlabel='Predicted', ylabel='True')

    plt.show()

def plot_feature_importance(series_fi, classifier_name, method_name):
    series_fi = series_fi.sort_values(ascending=False)
    series_fi.plot(kind='barh', title=f'Feature Importance - {classifier_name} - {method_name}')
    plt.xlabel("Importance")
    plt.show()

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


def create_result_output(classifier_name, method, average_precision, auc_roc, metric_mattews_coef,f1_score,accuracy, time_elapsed, testset_size):
    row_stats = {
        'Classifier': classifier_name,
        'Validation method': method,

        'Mattews Correlation Coef.': metric_mattews_coef,
        'F1-Score': f1_score,
        "Accuracy": accuracy,
        'Average Precision': average_precision,
        'AUCROC': auc_roc,
        'Training Time': time_elapsed,
        'testset_size': testset_size
    }
    return row_stats

def collect_matrics(y_true, y_pred, classifier_name, method_short_name, time_elapsed, testset_size):

    modelMetricsKfold = my_metrics.ModelMetrics(y_true, y_pred)
    metric_mattews_coef = modelMetricsKfold.matthews_corrcoef()
    f1_score = modelMetricsKfold.f1_score()
    accuracy = modelMetricsKfold.accuracy()

    average_precision = modelMetricsKfold.average_precision_score()
    auc_roc = modelMetricsKfold.auc_roc()

    row_results = create_result_output(classifier_name, method_short_name, average_precision, auc_roc, metric_mattews_coef, f1_score, accuracy,
                         time_elapsed, testset_size)
    return row_results


#Logistic Regression
def logreg_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model

def logreg_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def logistic_regression(df_result_metrics, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.log_reg.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    method_short_name = ValidationMethod.KFold.name

    logger.info(f"{classifier_name} - {method_name}")

    #kfold

    logreg = LogisticRegression(**_PARAMS_LORGREG)
    logreg, logreg_y_preds, logreg_y_trues, time_elapsed = model_predict_k_fold(logreg_train_func, logreg_pred_func, logreg)
    logreg_test_preds = logreg_pred_func(logreg, X_TEST)


    # collect the results - KFold

    # precision-recall curves (PRC)
    # Receiver-operator curves (ROC)
    plot_prc_curve(logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation PRC")
    plot_roc_curve(logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation ROC")

    plot_prc_curve(logreg_test_preds, Y_TEST, "LogReg Testing PRC")
    plot_roc_curve(logreg_test_preds, Y_TEST, "LogReg Testing ROC")

    # collect the result
    create_confusion_matrix(Y_TEST, logreg_test_preds.round(), classifier_name, method_short_name)

    metrics_logreg_kfold = collect_matrics(Y_TEST, logreg_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_logreg_kfold, ignore_index=True)

    fi_log_reg_kfold = pd.Series(logreg.coef_[0], index=X_TRAIN.columns)
    plot_feature_importance(fi_log_reg_kfold, classifier_name, method_short_name)

    row_features = {'classifier': "Logistic Regression"}
    for index, value in fi_log_reg_kfold.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    method_name = ValidationMethod.LOO.value
    method_short_name = ValidationMethod.LOO.name

    logger.info(f"{classifier_name} - {method_name}")

    logreg_loo = LogisticRegression(**_PARAMS_LORGREG)
    logreg_loo, logreg_y_preds_loo, logreg_y_trues_loo, time_elapsed = leave_one_out(logreg_train_func, logreg_pred_func, logreg_loo)
    logreg_loo_test_preds = logreg_pred_func(logreg_loo, X_TEST)


    print(classification_report(Y_TEST, logreg_loo_test_preds.round()))
    dict_report = classification_report(Y_TEST, logreg_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": classifier_name}
        if isinstance(value, dict):
            print(get_name_from_value(key))
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    fi_log_reg_loo = pd.Series(logreg_loo.coef_[0], index=X_TRAIN.columns)
    plot_feature_importance(fi_log_reg_loo, classifier_name, method_short_name)

    metrics_logreg_loo = collect_matrics(Y_TEST, logreg_loo_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_logreg_loo, ignore_index=True)

    return df_result_metrics, df_leave_one_out, df_feature_importance

# Random Forest
def rfc_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model

def rfc_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def random_forest(df_result_metrics, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.r_forest.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    method_short_name = ValidationMethod.KFold.name

    logger.info(f"{classifier_name} - {method_name}")

    rfc = RandomForestClassifier(**_PARAMS_RFC)
    rfc, rfc_y_preds, rfc_y_trues, time_elapsed = model_predict_k_fold(rfc_train_func, rfc_pred_func, rfc)
    rfc_test_preds = rfc_pred_func(rfc, X_TEST)


    plot_prc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Validation PRC")
    plot_roc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Vaidation ROC")

    plot_prc_curve(rfc_test_preds, Y_TEST, "RFC Testing PRC")
    plot_roc_curve(rfc_test_preds, Y_TEST, "RFC Testing ROC")

    plot_traning_curves(rfc_y_preds, rfc_y_trues)
    plot_testing_curves(rfc_test_preds, Y_TEST)

    create_confusion_matrix(Y_TEST, rfc_test_preds.round(), classifier_name, method_short_name)

    # collect the result
    metrics_rf_kfold = collect_matrics(Y_TEST, rfc_test_preds, classifier_name, method_name, time_elapsed, testset_size)

    df_result_metrics = df_result_metrics.append(metrics_rf_kfold, ignore_index=True)

    # get importance
    importance = rfc.feature_importances_
    fi_r_forest_kfold = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_r_forest_kfold, classifier_name, method_name)

    series_features = pd.Series(importance, index=X_TRAIN.columns)
    row_features = {'classifier': classifier_name}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out

    method_name = ValidationMethod.LOO.value
    method_short_name = ValidationMethod.LOO.name
    logger.info(f"{classifier_name} - {method_name}")

    rfc_loo = RandomForestClassifier(**_PARAMS_RFC)
    rfc_loo, rfc_y_preds_loo, rfc_y_trues_loo, time_elapsed = leave_one_out(rfc_train_func, rfc_pred_func, rfc_loo)
    rfc_loo_test_preds = rfc_pred_func(rfc_loo, X_TEST)

    print(classification_report(Y_TEST, rfc_loo_test_preds.round()))

    dict_report = classification_report(Y_TEST, rfc_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "Random Forest"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    importance = rfc_loo.feature_importances_
    fi_r_forest_kfold = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_r_forest_kfold, classifier_name, method_name)

    create_confusion_matrix(Y_TEST, rfc_loo_test_preds.round(), classifier_name, method_short_name)

    metrics_rf_loo = collect_matrics(Y_TEST, rfc_loo_test_preds, classifier_name, method_name, time_elapsed, testset_size)

    df_result_metrics = df_result_metrics.append(metrics_rf_loo, ignore_index=True)

    return df_result_metrics, df_leave_one_out, df_feature_importance

#Decision Tree

def dtc_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model


def dtc_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def decision_tree(df_result_metrics, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.d_tree.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    method_short_name = ValidationMethod.KFold.name
    logger.info(f"{classifier_name} - {method_name}")

    dtc = DecisionTreeClassifier(**_PARAMS_DTC)

    dtc, dtc_y_preds, dtc_y_trues, time_elapsed = model_predict_k_fold(dtc_train_func, dtc_pred_func, dtc)
    dtc_test_preds = dtc_pred_func(dtc, X_TEST)

    plot_prc_curve(dtc_y_preds, dtc_y_trues, "RFC Cross-Vaidation PRC")
    plot_roc_curve(dtc_y_preds, dtc_y_trues, "RFC Cross-Vaidation ROC")

    plot_prc_curve(dtc_test_preds, Y_TEST, "RFC Testing PRC")
    plot_roc_curve(dtc_test_preds, Y_TEST, "RFC Testing ROC")

    create_confusion_matrix(Y_TEST, dtc_test_preds.round(), classifier_name, method_short_name)

    # collect the result
    metrics_dt_kfold = collect_matrics(Y_TEST, dtc_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_dt_kfold, ignore_index=True)

    # get importance
    importance = dtc.feature_importances_

    fi_dtree_kfold = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_dtree_kfold, classifier_name, method_short_name)

    series_features = pd.Series(importance, index=X_TRAIN.columns)
    row_features = {'classifier': "Decision Tree"}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    method_name = ValidationMethod.LOO.value
    method_short_name = ValidationMethod.LOO.name
    logger.info(f"{classifier_name} - {method_name}")

    dtc_loo = DecisionTreeClassifier(**_PARAMS_DTC)
    dtc_loo, dtc_y_preds_loo, dtc_y_trues_loo, time_elapsed = leave_one_out(dtc_train_func, dtc_pred_func, dtc_loo)
    dtc_loo_test_preds = dtc_pred_func(dtc_loo, X_TEST)

    print(classification_report(Y_TEST, dtc_loo_test_preds.round()))

    # get importance
    importance = dtc_loo.feature_importances_

    fi_dtree_loo = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_dtree_loo, classifier_name, method_short_name)


    dict_report = classification_report(Y_TEST, dtc_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "Decision Tree"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    create_confusion_matrix(Y_TEST, dtc_test_preds.round(), classifier_name, method_short_name)

    metrics_dt_loo = collect_matrics(Y_TEST, dtc_loo_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_dt_loo, ignore_index=True)

    return df_result_metrics, df_leave_one_out, df_feature_importance


#XGBoost
def xgb_train_func(model, x_train, y_train, x_test, y_test):
    dtrain = xgb.DMatrix(data=x_train, label=y_train)

    bst = xgb.cv(_PARAMS_XGB,
        dtrain,
        num_boost_round=2000,
        nfold=5,
        early_stopping_rounds=200,
        verbose_eval=50
    )

    best_rounds = np.argmin(bst["test-logloss-mean"])
    bst = xgb.train(_PARAMS_XGB, dtrain, best_rounds)
    return bst

def xgb_pred_func(model, data):
    data = xgb.DMatrix(data=data)
    pred = model.predict(data)
    return pred

def xgboost(df_result_metrics, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.xgb.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    method_short_name = ValidationMethod.KFold.name
    logger.info(f"{classifier_name} - {method_name}")



    xgb_model, xgb_y_preds, xgb_y_trues, time_elapsed = model_predict_k_fold(xgb_train_func, xgb_pred_func)
    xgb_test_preds = xgb_pred_func(xgb_model, X_TEST)

    plot_prc_curve(xgb_y_preds, xgb_y_trues, "XGB Cross-Vaidation PRC")
    plot_roc_curve(xgb_y_preds, xgb_y_trues, "XGB Cross-Vaidation ROC")

    plot_prc_curve(xgb_test_preds, Y_TEST, "XGB Testing PRC")
    plot_roc_curve(xgb_test_preds, Y_TEST, "XGB Testing ROC")


    #kfold
    metrics_xgb_kfold = collect_matrics(Y_TEST, xgb_test_preds, classifier_name, method_name, time_elapsed, testset_size)

    df_result_metrics = df_result_metrics.append(metrics_xgb_kfold, ignore_index=True)

    # plot feature importance
    title = f"Feature Importance - {classifier_name} - {method_name}"
    plot_importance(xgb_model, title=title)
    #plt.title(f"Feature Importance - {classifier_name} - {method_name}")
    plt.show()

    series_features = pd.Series(xgb_model.feature_names, index=X_TRAIN.columns)
    row_features = {'classifier': "XGBoost"}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    method_name = ValidationMethod.LOO.value
    method_short_name = ValidationMethod.LOO.name

    xgb_loo, xgb_y_preds_loo, xgb_y_trues_loo, time_elapsed = leave_one_out(xgb_train_func, xgb_pred_func)
    xgb_loo_test_preds = xgb_pred_func(xgb_loo, X_TEST)

    print(classification_report(Y_TEST, xgb_loo_test_preds.round()))
    dict_report = classification_report(Y_TEST, xgb_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "XGBoost"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    # plot feature importance
    plot_importance(xgb_loo)
    plt.title(f"Feature Importance - {classifier_name} - {method_name}")
    plt.show()

    metrics_xgb_loo = collect_matrics(Y_TEST, xgb_loo_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_xgb_loo, ignore_index=True)

    return df_result_metrics, df_leave_one_out, df_feature_importance

#LightGBM
def gbm_train_func(model, x_train, y_train, x_test, y_test):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    gbm = lgb.train(
        _PARAMS_LIGHTGB,
        lgb_train,
        verbose_eval=True,
        num_boost_round=2000,
        valid_sets=lgb_eval,
        early_stopping_rounds=100
    )

    return gbm

def gbm_pred_func(model, data):
    return model.predict(data, num_iteration=model.best_iteration)

def light_gbm(df_result_metrics, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.lgbm.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    ################################# KFold
    method_name = ValidationMethod.KFold.value
    method_short_name = ValidationMethod.KFold.name
    logger.info(f"{classifier_name} - {method_name}")

    gbm, gbm_y_preds, gbm_y_trues, time_elapsed = model_predict_k_fold(gbm_train_func, gbm_pred_func)
    gbm_test_preds = gbm_pred_func(gbm, X_TEST)

    plot_prc_curve(gbm_y_preds, gbm_y_trues, "GBM Cross-Vaidation PRC")
    plot_roc_curve(gbm_y_preds, gbm_y_trues, "GBM Cross-Vaidation ROC")

    plot_prc_curve(gbm_test_preds, Y_TEST, "GBM Testing PRC")
    plot_roc_curve(gbm_test_preds, Y_TEST, "GBM Testing ROC")


    # collect the result
    create_confusion_matrix(Y_TEST, gbm_test_preds.round(), classifier_name, method_short_name)

    metrics_lgbm_kfold = collect_matrics(Y_TEST, gbm_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_lgbm_kfold, ignore_index=True)

    title = f"Feature Importance - {classifier_name} - {method_name}"
    ax = lgb.plot_importance(gbm, title=title) #, max_num_features=10
    #plt.title(f"Feature Importance - {classifier_name} - {method_name}")
    plt.show()

    df_feature_importance_2 = (
        pd.DataFrame({
            'feature': gbm.feature_name(),
            'importance': gbm.feature_importance(),
        })
            .sort_values('importance', ascending=False)
    )

    series_features = pd.Series(gbm.feature_importance(), index=gbm.feature_name())
    row_features = {'classifier': "LightGBM"}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    ################################# Leave one out
    method_name = ValidationMethod.LOO.value
    method_short_name = ValidationMethod.LOO.name
    logger.info(f"{classifier_name} - {method_name}")

    lgbm_loo, lgbm_y_preds_loo, lgbm_y_trues_loo, time_elapsed = leave_one_out(gbm_train_func, gbm_pred_func)
    lgbm_loo_test_preds = gbm_pred_func(lgbm_loo, X_TEST)

    print(classification_report(Y_TEST, lgbm_loo_test_preds.round()))

    dict_report = classification_report(Y_TEST, lgbm_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "LightGBM"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    create_confusion_matrix(Y_TEST, lgbm_loo_test_preds.round(), classifier_name, method_short_name)

    print('Plotting feature importances...')
    title = (f"Feature Importance - {classifier_name} - {method_name}")
    ax = lgb.plot_importance(lgbm_loo, title=title) #, max_num_features=10
    #plt.title(f"Feature Importance - {classifier_name} - {method_name}")
    plt.show()

    metrics_lgbm_kfold = collect_matrics(Y_TEST, lgbm_loo_test_preds, classifier_name, method_name, time_elapsed, testset_size)
    df_result_metrics = df_result_metrics.append(metrics_lgbm_kfold, ignore_index=True)

    return df_result_metrics, df_leave_one_out, df_feature_importance


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
    logger = log_configuration.logger
    logger.info("Script started...")

    show_graphs = True

    config_params = {}
    config_params['show_confusion_matrix'] = True
    config_params['show_roc'] = True

    run_hyper_tuning = False
    run_models = False
    read_result_df_saved = True
    check_options(run_hyper_tuning, run_models, read_result_df_saved)

    if run_hyper_tuning:
        logger.info("Tuning models...")
        #hyper_tuning(model, param_grid)
        tuning.run_tuning(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        exit()

    # Data Frame to collect all results of the classifiers
    df_result_metrics = pd.DataFrame()
    df_feature_importance = pd.DataFrame()
    df_leave_one_out = pd.DataFrame()

    #todo create class to manage the changing between the datasets and file names below
    # DF saved files names
    #result_filename = "all_classifiers_provided_paper_features"
    #result_filename = "all_classifiers_new_features"
    result_filename = "all_classifiers_reproduced_paper_features"
    #result_filename = "single_classifier"


    if run_models:
        logger.info("Run models...")
        df_result_metrics, df_leave_one_out, df_feature_importance = logistic_regression(df_result_metrics,  df_leave_one_out, df_feature_importance)

        df_result_metrics, df_leave_one_out, df_feature_importance = decision_tree(df_result_metrics, df_leave_one_out, df_feature_importance)
        df_result_metrics, df_leave_one_out, df_feature_importance = random_forest(df_result_metrics, df_leave_one_out,df_feature_importance)

        df_result_metrics, df_leave_one_out, df_feature_importance = xgboost(df_result_metrics, df_leave_one_out, df_feature_importance)
        df_result_metrics, df_leave_one_out, df_feature_importance = light_gbm(df_result_metrics, df_leave_one_out, df_feature_importance)


        save_dataframe(df_result_metrics, result_filename)

        try:
            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.KFold.value, 'F1-Score')
            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.LOO.value, 'F1-Score')

            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.KFold.value, 'Mattews Correlation Coef.')
            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.LOO.value, 'Mattews Correlation Coef.')

            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.KFold.value, 'Accuracy')
            results_plot.create_plot_result_ml(df_result_metrics, ValidationMethod.LOO.value, 'Accuracy')
        except:
            print("Something went wrong. Maybe only one model was chosen")

    if read_result_df_saved:
        df_result_metrics = load_dataframe(result_filename)


    #results_plot.create_plot_result_training_time(df_result_metrics)
    print(df_result_metrics)
    print(tabulate(df_result_metrics, headers='keys', tablefmt='psql'))
    print(tabulate(df_leave_one_out, headers='keys', tablefmt='psql'))
    print(tabulate(df_feature_importance, headers='keys', tablefmt='psql'))

    report_outputs = True
    if report_outputs:
        #export result as table image
        results_plot.create_table_result(df_result_metrics, ValidationMethod.KFold.value, testset_size, result_filename)
        results_plot.create_table_result(df_result_metrics, ValidationMethod.LOO.value, testset_size, result_filename)

        #time
        results_plot.create_table_result_time_exec(df_result_metrics, 'Leave-One-Out', result_filename)
        results_plot.create_table_result_time_exec(df_result_metrics, 'K-Fold Cross-Validation', result_filename)
