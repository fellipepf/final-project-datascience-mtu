
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import tsfel as tsfel

sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py
import utils
from psykose import LoadDataset




def get_serie():
    control, patient = LoadDataset().get_dataset_by_class()
    print("")
    id = "control_5"
    df_serie = control[id]["timeserie"]

    df_serie["datetime"] = pd.to_datetime(df_serie["timestamp"])

    values = df_serie["activity"].values
    return values

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

if __name__ == '__main__':

    serie = get_serie()
    # If no argument is passed retrieves all available features
    cfg_file = tsfel.get_features_by_domain()

    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, window_size=480)

    print_full(X_train)

