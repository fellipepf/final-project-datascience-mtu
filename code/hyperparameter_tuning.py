
import sys
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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

    xgb_param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror', "binary:logistic"]
    }
    #Best: 0.865714 using {'subsample': 0.5, 'objective': 'reg:squarederror', 'n_estimators': 500, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

    params_dic[Classifiers.XGB.name] = xgb_param_tuning

    xgb_skt_param_grid = {
        'n_estimators': [5, 10, 50, 100, 250, 500,1000],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'subsample': [0.5, 0.7, 1.0],
        'max_depth': [3,5, 7, 9],
        'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 5, endpoint=True ),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [3,5,10,12]
    }
    #gridseach
    #Best: 0.885154 using {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
    #random search
    #Best: 0.839608 using {'subsample': 0.7, 'n_estimators': 500, 'min_weight_fraction_leaf': 0.375, 'max_leaf_nodes': 12, 'max_features': 'auto', 'max_depth': 7, 'learning_rate': 0.01}
    #Best: 0.863922 using {'subsample': 1.0, 'n_estimators': 250, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': 3, 'max_features': 'log2', 'max_depth': 7, 'learning_rate': 1}

    params_dic[Classifiers.XGB_skt.name] = xgb_skt_param_grid

    lgbm_params = {
        'bagging_fraction': (0.5, 0.8),
        'bagging_freq': (5, 8),
        "learning_rate": (0.01, 0.1, 1),
        'feature_fraction': (0.5, 0.8),
        'max_depth': (5, 10, 13),
        'min_data_in_leaf': (90, 120),
        'num_leaves': (10, 20,30),
        'colsample_bytree': (0.01, 1.0, 'uniform'),  # enabler of bagging fraction
        'min_child_weight': (0, 10),  # minimal number of data in one leaf.
        'subsample_for_bin': (100000, 500000),  # number of data that sampled for histogram bins
    }
    #Best: 0.849300 using {'bagging_fraction': 0.8, 'bagging_freq': 8, 'feature_fraction': 0.5, 'max_depth': 10, 'min_data_in_leaf': 90, 'num_leaves': 1200}
#random
    #Best: 0.846639 using {'subsample_for_bin': 500000, 'num_leaves': 10, 'min_data_in_leaf': 90, 'min_child_weight': 10, 'max_depth': 13, 'learning_rate': 1, 'feature_fraction': 0.5, 'colsample_bytree': 0.01, 'bagging_freq': 5, 'bagging_fraction': 0.8}

    params_dic[Classifiers.LGBM.name] = lgbm_params

    return params_dic

def define_classifiers():
    classifiers_dic = {}

    classifiers_dic[Classifiers.LR.name] = LogisticRegression()
    classifiers_dic[Classifiers.RF.name] = RandomForestClassifier()
    classifiers_dic[Classifiers.DT.name] = DecisionTreeClassifier()
    classifiers_dic[Classifiers.XGB.name] = xgb.XGBClassifier()
    classifiers_dic[Classifiers.XGB_skt.name] = GradientBoostingClassifier()
    classifiers_dic[Classifiers.LGBM.name] = lgb.LGBMClassifier()

    return classifiers_dic

def hyper_tuning(model, grid_params, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST):
    scores = [
        # 'precision',
        'recall',
        # 'f1'
    ]

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #search_method = GridSearchCV(estimator=model, param_grid=grid_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, verbose=2)
    search_method = RandomizedSearchCV(estimator=model, param_distributions=grid_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, n_iter=15, verbose=2)
    grid_result = search_method.fit(X_TRAIN, Y_TRAIN)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    best_grid = search_method.best_estimator_
    grid_accuracy = evaluate(best_grid, X_TEST, Y_TEST)

    print(grid_accuracy)

def run_tuning(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST):
    classifiers = define_classifiers()
    params = define_params()

    #select classifiers to be tuned
    classifier = Classifiers.LGBM.name

    hyper_tuning(classifiers[classifier], params[classifier], X_TRAIN, X_TEST, Y_TRAIN, Y_TEST )

if __name__ == '__main__':
    #define_params()
    print(Classifiers.LR.name)
    print(Classifiers.LR.value)
    print(Classifiers.RF)