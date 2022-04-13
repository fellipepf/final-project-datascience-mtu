
from psykose_dataset import LoadDataset, PreProcessing
import sys

import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt, dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
import dataframe_image as dfi

def boxplot():
    pass

def lineplot():
    pass

def perform_eda_of_person(id):
    pass

def plot_line_graph(df_time_period, id):

    values = df_time_period[id]

    list_df_morning = values['morning']
    merged = pd.concat(list_df_morning)

    sns.set(rc={'figure.figsize': (11, 4)})
    merged['activity'].plot(linewidth=0.5);
    plt.show()


if __name__ == '__main__':
    #load
    loadDataset = LoadDataset()
    control, patient = loadDataset.get_dataset_by_class()
    control_patient = loadDataset.get_dataset_joined()

    data_process = PreProcessing(control, patient)

    id_person = ["control_12"]
    plot_line_graph(data_process)