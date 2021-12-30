
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


import lightgbm as lgb

import neptune
#import skopt

import enum

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

class Classifiers(enum.Enum):
   LR = "Logistic Regression"
   RF = "Random Forest"
   DT = "Decision Tree"
   XGB = "XGBoost"
   XGB_skt = "XGBoost_skt"  #scikit learn implementation of XGBoost
   LGBM = "LightGBM"


def define_params():
    params_dic = {}

    lr_param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }

    params_dic[Classifiers.LR.name] = lr_param_grid

    rf_param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    params_dic[Classifiers.RF.name] = rf_param_grid

    dt_param_grid = {
        #'n_estimators': [10, 100, 200, 300, 1000],
        'criterion': ['gini', 'entropy'],
        "max_features": ['auto', 'sqrt', 'log2'],
        "max_depth": [2, 4, 6, 8, 10],
        'min_samples_split': [2,4,6], #default =2
        "min_samples_leaf": [1,2,4],  #default=1
        "min_weight_fraction_leaf": np.linspace(0.0, 0.5, 5, endpoint=True ),  #values between 0.0 and 0.5 - value 5 is discarted
        "max_leaf_nodes": [3,5,10,12]

    }
    #Best: 0.824090 using {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 12, 'min_samples_leaf': 4, 'min_samples_split': 6, 'min_weight_fraction_leaf': 0.0}

    params_dic[Classifiers.DT.name] = dt_param_grid

    return params_dic

def define_classifiers():
    classifiers_dic = {}

    classifiers_dic[Classifiers.LR.name] = LogisticRegression()
    classifiers_dic[Classifiers.RF.name] = RandomForestClassifier()
    classifiers_dic[Classifiers.DT.name] = DecisionTreeClassifier()

    return classifiers_dic

def hyper_tuning(model, grid_params, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST):

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

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_TEST, Y_TEST)

def run_tuning(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST):
    classifiers = define_classifiers()
    params = define_params()

    #select classifiers to be tuned
    classifier = Classifiers.DT.name


    xgb_model = GradientBoostingClassifier()
#    light_gbm = lgb()


    hyper_tuning(classifiers[classifier], params[classifier], X_TRAIN, X_TEST, Y_TRAIN, Y_TEST )

if __name__ == '__main__':
    #define_params()
    print(Classifiers.LR.name)
    print(Classifiers.LR.value)
    print(Classifiers.RF)