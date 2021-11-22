from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

from matplotlib import pyplot
from xgboost import plot_importance

color = sns.color_palette()



PATH_TO_FILE = "../../psykose/schizophrenia-features.csv"


_PARAMS_LORGREG = {
    "penalty": "l2", "C": 1.0, "class_weight": "balanced",
    "random_state": 2018, "solver": "liblinear", "n_jobs": 1
}

_PARAMS_RFC = {
    "n_estimators": 10,
    "max_features": "auto", "max_depth": None,
    "min_samples_split": 2, "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None, "bootstrap": True,
    "oob_score": False, "n_jobs": -1, "random_state": 2018,
    "class_weight": "balanced"
}

_PARAMS_XGB = {
    "nthread":16, "learning_rate": 0.3, "gamma": 0, "max_depth": 6, "verbosity": 0,
    "min_child_weight": 1, "max_delta_step": 0, "subsample": 1.0, "colsample_bytree": 1.0,
    "objective":"binary:logistic", "num_class":1, "eval_metric":"logloss", "seed":2018,
}

_PARAMS_LIGHTGB = {
    "task": "train", "num_class":1, "boosting": "gbdt", "verbosity": -1,
    "objective": "binary", "metric": "binary_logloss", "metric_freq":50, "is_training_metric":False,
    "max_depth":4, "num_leaves": 31, "learning_rate": 0.01, "feature_fraction": 1.0, "bagging_fraction": 1.0,
    "bagging_freq": 0, "bagging_seed": 2018, "num_threads":16
}



#####

data = pd.read_csv(PATH_TO_FILE)

dataX = data.copy().drop(["class", "class_str", "userid"], axis=1)
dataY = data["class"].copy()

scaler = pp.StandardScaler(copy=True)

dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])


testset_size = 0.5

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    dataX,
    dataY,
    test_size=testset_size,
    random_state=2019,
    stratify=dataY
)

#####


def plot_prc_curve(y_preds, y_trues, title=None):
    precision, recall, _ = metrics.precision_recall_curve(
        y_trues,
        y_preds
    )

    average_precision = metrics.average_precision_score(
        y_trues,
        y_preds
    )

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


#######
'''
Logistic Regression
'''

def logreg_train_func(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model

def logreg_pred_func(model, data):
    return model.predict_proba(data)[:, 1]

def logistic_regression():

    logreg = LogisticRegression( **_PARAMS_LORGREG )

    logreg, logreg_y_preds, logreg_y_trues = model_predict_k_fold( logreg_train_func, logreg_pred_func, logreg )
    logreg_test_preds = logreg_pred_func( logreg, X_TEST )

    #precision-recall curves (PRC)
    #Receiver-operator curves (ROC)
    average_precision = plot_prc_curve( logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation PRC" )
    auc_roc           = plot_roc_curve( logreg_y_preds, logreg_y_trues, "LogReg Cross-Vaidation ROC" )

    plot_prc_curve( logreg_test_preds, Y_TEST, "LogReg Testing PRC" )
    plot_roc_curve( logreg_test_preds, Y_TEST, "LogReg Testing ROC" )


    #collect the result
    row_stats = {'classifier': "Logistic Regression",
                 'Average Precision': average_precision,
                 'AUCROC': auc_roc,
                 'testset_size': testset_size}
    df_result.append(row_stats, ignore_index=True)

    # get importance
    importance = logreg.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
	    print('Feature: %0d, Score: %.5f' % (i,v))

    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    (pd.Series(logreg.coef_[0], index=X_TRAIN.columns)
        .nlargest(4)
        .plot(kind='barh'))

    series_features = pd.Series(logreg.coef_[0], index=X_TRAIN.columns)
    row_features = {'classifier': "Logistic Regression"}
    for index, value in series_features.items():
        new_values =  {index: value}
        row_features.update(new_values)
    df_feature_importance.append(row_features, ignore_index=True)



if __name__ == '__main__':
    # Data Frame to collect all results of the classifiers
    df_result = pd.DataFrame()
    df_feature_importance = pd.DataFrame()

    logistic_regression()
    print("")