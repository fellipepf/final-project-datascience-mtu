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
from IPython.display import display, HTML
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

color = sns.color_palette()

sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py
import utils
from utils import get_name_from_value
from utils import Target

# PATH_TO_FILE = "../../psykose/schizophrenia-features.csv"

# my baseline features
PATH_TO_FILE = "baseline_time_period.csv"

penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
class_weight='balanced'
solver = ['liblinear', 'saga']


_PARAMS_LORGREG = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 2018,
    "solver": "liblinear",
    "n_jobs": 1
}

_PARAMS_RFC = {
    "n_estimators": 10,
    "max_features": "auto",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": -1,
    "random_state": 2018,
    "class_weight": "balanced"
}

_PARAMS_DTC = {

    "max_features": "auto",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None,
    "random_state": 2018,
    "class_weight": "balanced"

}

_PARAMS_XGB = {
    "nthread": 16, "learning_rate": 0.3, "gamma": 0, "max_depth": 6, "verbosity": 0,
    "min_child_weight": 1, "max_delta_step": 0, "subsample": 1.0, "colsample_bytree": 1.0,
    "objective": "binary:logistic", "num_class": 1, "eval_metric": "logloss", "seed": 2018,
}

_PARAMS_LIGHTGB = {
    "task": "train", "num_class": 1, "boosting": "gbdt", "verbosity": -1,
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


testset_size = 0.2

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


def model_predict_k_fold(train_func, pred_func, model=None, n_splits=10, shuffle=True, random_state=2018):
    y_preds = []
    y_trues = []

    k_fold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for train_index, fold_index in k_fold.split(np.zeros(len(X_TRAIN)), Y_TRAIN.ravel()):
        x_fold_train, x_fold_test = X_TRAIN.iloc[train_index, :], X_TRAIN.iloc[fold_index, :]
        y_fold_train, y_fold_test = Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[fold_index]

        model = train_func(model, x_fold_train, y_fold_train, x_fold_test, y_fold_test)
        y_pred = pred_func(model, x_fold_test)

        y_preds.extend(list(y_pred))
        y_trues.extend(list(y_fold_test))

    return model, np.array(y_preds), np.array(y_trues)


def leave_one_out(train_func, pred_func, model=None):
    y_preds = []
    y_trues = []

    loo = LeaveOneOut()

    for train_index, fold_index in loo.split(np.zeros(len(X_TRAIN)), Y_TRAIN.ravel()):
        x_fold_train, x_fold_test = X_TRAIN.iloc[train_index, :], X_TRAIN.iloc[fold_index, :]
        y_fold_train, y_fold_test = Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[fold_index]

        model = train_func(model, x_fold_train, y_fold_train, x_fold_test, y_fold_test)
        y_pred = pred_func(model, x_fold_test)

        y_preds.extend(list(y_pred))
        y_trues.extend(list(y_fold_test))

    return model, np.array(y_preds), np.array(y_trues)

def create_confusion_matrix(y_true, y_preds, classifier_name=None):
    cm = confusion_matrix(y_true, y_preds) #, normalize='all'
    cmd = ConfusionMatrixDisplay(cm, display_labels=['healthy', 'patient'])
    cmd = cmd.plot(cmap=plt.cm.Blues, values_format='g')
    cmd.ax_.set_title(f'Confusion Matrix - {classifier_name}')
    cmd.plot()
    #cmd.ax_.set(xlabel='Predicted', ylabel='True')

    plt.show()

def plot_feature_importance(series_fi, classifier_name):
    series_fi = series_fi.sort_values(ascending=False)
    series_fi.plot(kind='barh', title=f'Feature Importance - {classifier_name}')
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


def create_result_output(classifier_name, method, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size):
    row_stats = {
        'Classifier': classifier_name,
        'Validation method': method,

        'Mattews Correlation Coef': metric_mattews_coef,
        'F1-Score': f1_score,
        'Average Precision': average_precision,
        'AUCROC': auc_roc,
        'testset_size': testset_size
    }
    return row_stats

#Logistic Regression
def logreg_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model

def logreg_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def logistic_regression(df_result_kfold, df_leave_one_out, df_feature_importance):
    logger = log_configuration.logger
    logger.info("Logistic Regression...")

    classifier_name = "Logistic Regression"

    #kfold
    method_name = ValidationMethod.KFold.value
    logger.info(f"{classifier_name} - {method_name}")
    logreg = LogisticRegression(**_PARAMS_LORGREG)

    logreg, logreg_y_preds, logreg_y_trues = model_predict_k_fold(logreg_train_func, logreg_pred_func, logreg)
    logreg_test_preds = logreg_pred_func(logreg, X_TEST)

    create_confusion_matrix(Y_TEST, logreg_test_preds.round(), classifier_name)

    # precision-recall curves (PRC)
    # Receiver-operator curves (ROC)
    average_precision = plot_prc_curve(logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation PRC")
    auc_roc = plot_roc_curve(logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation ROC")

    plot_prc_curve(logreg_test_preds, Y_TEST, "LogReg Testing PRC")
    plot_roc_curve(logreg_test_preds, Y_TEST, "LogReg Testing ROC")



    # collect the results - KFold
    metrics_kfold = my_metrics.ModelMetrics(Y_TEST, logreg_test_preds.round()).all_metrics()

    metric_mattews_coef = metrics_kfold['matthews_corrcoef']
    f1_score = metrics_kfold['f1_score']

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    fi_log_reg_kfold = pd.Series(logreg.coef_[0], index=X_TRAIN.columns)
    plot_feature_importance(fi_log_reg_kfold, classifier_name)

    row_features = {'classifier': "Logistic Regression"}
    for index, value in fi_log_reg_kfold.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    logger.info("Logistic Regression - Leave-One-Out")
    method_name = "Leave-One-Out"

    logreg_loo = LogisticRegression(**_PARAMS_LORGREG)
    logreg_loo, logreg_y_preds_loo, logreg_y_trues_loo = leave_one_out(logreg_train_func, logreg_pred_func, logreg_loo)
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

    modelMetricsLOO = my_metrics.ModelMetrics(Y_TEST, logreg_loo_test_preds.round())
    metric_mattews_coef = modelMetricsLOO.matthews_corrcoef()
    f1_score = modelMetricsLOO.f1_score()

    row_stats = create_result_output(classifier_name,method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)



    return df_result_kfold, df_leave_one_out, df_feature_importance

# Random Forest
def rfc_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model

def rfc_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def random_forest(df_result_kfold, df_leave_one_out, df_feature_importance):
    classifier_name = "Random Forest"
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    logger.info(f"{classifier_name} - {method_name}")

    rfc = RandomForestClassifier(**_PARAMS_RFC)

    rfc, rfc_y_preds, rfc_y_trues = model_predict_k_fold(rfc_train_func, rfc_pred_func, rfc)
    rfc_test_preds = rfc_pred_func(rfc, X_TEST)


    average_precision = plot_prc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Validation PRC")
    auc_roc = plot_roc_curve(rfc_y_preds, rfc_y_trues, "RFC Cross-Vaidation ROC")

    plot_prc_curve(rfc_test_preds, Y_TEST, "RFC Testing PRC")
    plot_roc_curve(rfc_test_preds, Y_TEST, "RFC Testing ROC")

    metric_mattews_coef = matthews_corrcoef(Y_TEST, rfc_test_preds.round())
    f1_score = metrics.f1_score(Y_TEST, rfc_test_preds.round())

    # collect the result
    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    # get importance
    importance = rfc.feature_importances_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    #
    # # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()
    #
    # #
    # (pd.Series(importance, index=X_TRAIN.columns)
    #  # .nlargest(4)
    #  .plot(kind='barh'))

    fi_r_forest_kfold = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_r_forest_kfold, classifier_name)

    series_features = pd.Series(importance, index=X_TRAIN.columns)
    row_features = {'classifier': classifier_name}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    method_name = ValidationMethod.LOO.name
    rfc_loo = RandomForestClassifier(**_PARAMS_RFC)
    rfc_loo, rfc_y_preds_loo, rfc_y_trues_loo = leave_one_out(rfc_train_func, rfc_pred_func, rfc_loo)
    rfc_loo_test_preds = logreg_pred_func(rfc_loo, X_TEST)

    print(classification_report(Y_TEST, rfc_loo_test_preds.round()))

    dict_report = classification_report(Y_TEST, rfc_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "Random Forest"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    modelMetricsLOO = my_metrics.ModelMetrics(Y_TEST, rfc_loo_test_preds.round())
    metric_mattews_coef = modelMetricsLOO.matthews_corrcoef()
    f1_score = modelMetricsLOO.f1_score()

    row_stats = create_result_output(classifier_name,method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    return df_result_kfold, df_leave_one_out, df_feature_importance

#Decision Tree

def dtc_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model


def dtc_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def decision_tree(df_result_kfold, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.d_tree.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    logger.info(f"{classifier_name} - {method_name}")


    dtc = DecisionTreeClassifier(**_PARAMS_DTC)

    dtc, dtc_y_preds, dtc_y_trues = model_predict_k_fold(dtc_train_func, dtc_pred_func, dtc)
    dtc_test_preds = dtc_pred_func(dtc, X_TEST)


    average_precision = plot_prc_curve(dtc_y_preds, dtc_y_trues, "RFC Cross-Vaidation PRC")
    auc_roc = plot_roc_curve(dtc_y_preds, dtc_y_trues, "RFC Cross-Vaidation ROC")

    plot_prc_curve(dtc_test_preds, Y_TEST, "RFC Testing PRC")
    plot_roc_curve(dtc_test_preds, Y_TEST, "RFC Testing ROC")

    # collect the result
    metric_mattews_coef = matthews_corrcoef(Y_TEST, dtc_test_preds.round())
    f1_score = metrics.f1_score(Y_TEST, dtc_test_preds.round())

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    # get importance
    importance = dtc.feature_importances_
    # # summarize feature importance
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    #
    # # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()



    fi_dtree_kfold = pd.Series(importance, index=X_TRAIN.columns)
    plot_feature_importance(fi_dtree_kfold, classifier_name)

    series_features = pd.Series(importance, index=X_TRAIN.columns)
    row_features = {'classifier': "Decision Tree"}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out
    method_name = ValidationMethod.LOO.name
    dtc_loo = RandomForestClassifier(**_PARAMS_RFC)
    dtc_loo, dtc_y_preds_loo, dtc_y_trues_loo = leave_one_out(dtc_train_func, dtc_pred_func, dtc_loo)
    dtc_loo_test_preds = logreg_pred_func(dtc_loo, X_TEST)

    print(classification_report(Y_TEST, dtc_loo_test_preds.round()))

    dict_report = classification_report(Y_TEST, dtc_loo_test_preds.round(), output_dict=True)
    for key, value in dict_report.items():
        row_loo = {"Classifier": "Decision Tree"}
        if isinstance(value, dict):
            row_loo["Class"] = get_name_from_value(key)
            row_loo.update(value)

            # collect the result for leave one out
            df_leave_one_out = df_leave_one_out.append(row_loo, ignore_index=True)

    modelMetricsLOO = my_metrics.ModelMetrics(Y_TEST, dtc_loo_test_preds.round())
    metric_mattews_coef = modelMetricsLOO.matthews_corrcoef()
    f1_score = modelMetricsLOO.f1_score()

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,
                                     f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    return df_result_kfold, df_leave_one_out, df_feature_importance


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

def xgboost(df_result_kfold, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.xgb.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    logger.info(f"{classifier_name} - {method_name}")



    xgb_model, xgb_y_preds, xgb_y_trues = model_predict_k_fold(xgb_train_func, xgb_pred_func)
    xgb_test_preds = xgb_pred_func(xgb_model, X_TEST)

    average_precision = plot_prc_curve(xgb_y_preds, xgb_y_trues, "XGB Cross-Vaidation PRC")
    auc_roc = plot_roc_curve(xgb_y_preds, xgb_y_trues, "XGB Cross-Vaidation ROC")

    plot_prc_curve(xgb_test_preds, Y_TEST, "XGB Testing PRC")
    plot_roc_curve(xgb_test_preds, Y_TEST, "XGB Testing ROC")


    #kfold
    metric_mattews_coef = matthews_corrcoef(Y_TEST, xgb_test_preds.round())
    f1_score = metrics.f1_score(Y_TEST, xgb_test_preds.round())

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    # plot feature importance
    plot_importance(xgb_model)
    plt.show()

    series_features = pd.Series(xgb_model.feature_names, index=X_TRAIN.columns)
    row_features = {'classifier': "XGBoost"}
    for index, value in series_features.items():
        new_values = {index: value}
        row_features.update(new_values)
    df_feature_importance = df_feature_importance.append(row_features, ignore_index=True)

    # Leave one out

    xgb_loo, xgb_y_preds_loo, xgb_y_trues_loo = leave_one_out(xgb_train_func, xgb_pred_func)
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

    modelMetricsLOO = my_metrics.ModelMetrics(Y_TEST, xgb_loo_test_preds.round())
    metric_mattews_coef = modelMetricsLOO.matthews_corrcoef()
    f1_score = modelMetricsLOO.f1_score()

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,
                                     f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)


    return df_result_kfold, df_leave_one_out, df_feature_importance

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

def light_gbm(df_result_kfold, df_leave_one_out, df_feature_importance):
    classifier_name = Classifier.lgbm.value
    logger = log_configuration.logger
    logger.info(f"{classifier_name}...")

    method_name = ValidationMethod.KFold.value
    logger.info(f"{classifier_name} - {method_name}")



    gbm, gbm_y_preds, gbm_y_trues = model_predict_k_fold(gbm_train_func, gbm_pred_func)
    gbm_test_preds = gbm_pred_func(gbm, X_TEST)

    average_precision = plot_prc_curve(gbm_y_preds, gbm_y_trues, "GBM Cross-Vaidation PRC")
    auc_roc = plot_roc_curve(gbm_y_preds, gbm_y_trues, "GBM Cross-Vaidation ROC")

    plot_prc_curve(gbm_test_preds, Y_TEST, "GBM Testing PRC")
    plot_roc_curve(gbm_test_preds, Y_TEST, "GBM Testing ROC")


    # collect the result
    metric_mattews_coef = matthews_corrcoef(Y_TEST, gbm_test_preds.round())
    f1_score = metrics.f1_score(Y_TEST, gbm_test_preds.round())

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)

    print('Plotting feature importances...')
    ax = lgb.plot_importance(gbm, max_num_features=10)
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

    # Leave one out
    method_name = ValidationMethod.LOO.name
    logger.info(f"{classifier_name} - {method_name}")

    lgbm_loo, lgbm_y_preds_loo, lgbm_y_trues_loo = leave_one_out(gbm_train_func, gbm_pred_func)
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

    modelMetricsLOO = my_metrics.ModelMetrics(Y_TEST, lgbm_loo_test_preds.round())
    metric_mattews_coef = modelMetricsLOO.matthews_corrcoef()
    f1_score = modelMetricsLOO.f1_score()

    row_stats = create_result_output(classifier_name, method_name, average_precision, auc_roc, metric_mattews_coef,
                                     f1_score, testset_size)
    df_result_kfold = df_result_kfold.append(row_stats, ignore_index=True)


    return df_result_kfold, df_leave_one_out, df_feature_importance

def testing(logReg):
    predictionsTestSetLogisticRegression = pd.DataFrame(data=[], index=Y_TEST.index, columns=['prediction'])
    predictionsTestSetLogisticRegression.loc[:, 'prediction'] = logReg.predict_proba(X_TEST)[:, 1]
    logLossTestSetLogisticRegression = log_loss(Y_TEST, predictionsTestSetLogisticRegression)


def hyper_tuning_random_forest():
    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    #X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    grid_search.fit(X_TRAIN, Y_TRAIN)

    print(grid_search.best_params_)

def hyper_tuning(model, grid_params ):

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, verbose=2)
    grid_result = grid_search.fit(X_TRAIN, Y_TRAIN)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


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

if __name__ == '__main__':
    logger = log_configuration.logger
    logger.info("Script started...")

    show_graphs = True

    config_params = {

    }

    run_hyper_tuning = False
    run_models = True
    check_options(run_hyper_tuning, run_models)

    if run_hyper_tuning:
        #hyper_tuning(model, param_grid)
        tuning.run_tuning(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        exit()

    # Data Frame to collect all results of the classifiers
    df_result_kfold = pd.DataFrame()
    df_feature_importance = pd.DataFrame()
    df_leave_one_out = pd.DataFrame()

    if run_models:
        df_result_kfold, df_leave_one_out, df_feature_importance = logistic_regression(df_result_kfold,  df_leave_one_out, df_feature_importance)
        df_result_kfold, df_leave_one_out, df_feature_importance = random_forest(df_result_kfold, df_leave_one_out,df_feature_importance)
        df_result_kfold, df_leave_one_out, df_feature_importance = decision_tree(df_result_kfold, df_leave_one_out, df_feature_importance)
        #df_result_kfold, df_leave_one_out, df_feature_importance = xgboost(df_result_kfold, df_leave_one_out, df_feature_importance)
        #df_result_kfold, df_leave_one_out, df_feature_importance = light_gbm(df_result_kfold, df_leave_one_out, df_feature_importance)

        results_plot.create_plot_result_ml(df_result_kfold, ValidationMethod.KFold.value, 'F1-Score')
        results_plot.create_plot_result_ml(df_result_kfold, ValidationMethod.LOO.value, 'F1-Score')


    print(df_result_kfold)
    print(tabulate(df_result_kfold, headers='keys', tablefmt='psql'))
    print(tabulate(df_leave_one_out, headers='keys', tablefmt='psql'))
    print(tabulate(df_feature_importance, headers='keys', tablefmt='psql'))
    #dfi.export(df_result_kfold, 'output_images/dataframe.png')
