# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys

import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt, dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
from matplotlib.dates import DateFormatter, date2num
import dataframe_image as dfi


def create_plot_result_ml(df_result, method, metric):
    method_values = df_result['Validation method'] == method
    df_method = df_result[method_values]
    df_method.sort_values(metric,ascending=False, inplace=True)

    ax = sns.barplot(x='Classifier', y=metric, data=df_method, ci=None)
    ax.bar_label(ax.containers[0])
    ax.set_title(f'{method}')

    # Show the plot
    plt.show()

def create_plot_result_training_time(df_result):
    #df_result['Training Time'] = df_result["Training Time"].values.astype('timedelta64[ms]').plot.hist()
    df_result['Training Time'] = pd.to_datetime(df_result['Training Time'])
    #ax = sns.barplot(data=df_result, x='Classifier', y='Training Time', hue='Validation method')

    fig, ax = plt.subplots()

    myFmt = DateFormatter("%H:%M:%S")
    ax.yaxis.set_major_formatter(myFmt)

    ax.plot(df_result['Classifier'], df_result['Training Time'])

    plt.gcf().autofmt_xdate()

    plt.show()
    plt.show()

def df_style(val):
    return "font-weight: bold"


def create_table_result(df_result, method, testset_size, result_filename):
    method_values = df_result['Validation method'] == method
    df_method = df_result[method_values]
    df_method = df_method.sort_values(by=['Average Precision'], ascending=False).reset_index(drop=True)
    df_method['Test set size'] = testset_size * 100
    df_method['Test set size'] = df_method['Test set size'].astype(int)
    df_method = df_method.rename(columns={'Mattews Correlation Coef.': 'MCC'})
    df_method = df_method[['Classifier','Validation method', 'Test set size', 'Average Precision', 'AUCROC', 'Accuracy', 'F1-Score', 'MCC']]

    first_row = pd.IndexSlice[0]
    df_method = df_method.style.applymap(df_style, subset=first_row)
    dfi.export(df_method, f'output_images/df_result_{result_filename}_{method}.png')

def create_table_result_time_exec(df_result, method, result_filename):
    method_values = df_result['Validation method'] == method
    df_method = df_result[method_values]

    time_filter = df_method[['Classifier','Validation method', 'Training Time']]
    #df_method = df_result[time_filter]
    time_filter = pd.DataFrame(time_filter)
    time_filter = time_filter.sort_values(by=['Training Time'], ascending=False).reset_index(drop=True)


    #df_method = df_method[['Classifier','Validation method', 'Test set size', 'Average Precision', 'AUCROC', 'Accuracy', 'F1-Score', 'MCC']]

    #first_row = pd.IndexSlice[0]
    #df_method = df_method.style.applymap(df_style, subset=first_row)
    time_filter = time_filter.rename(columns={'Training Time': 'Training Time (hh:mm:ss.ms)'})
    dfi.export(time_filter, f'output_images/df_result_time_exec{result_filename}_{method}.png')