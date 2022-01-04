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



def create_plot_result_ml(df_result, method, metric):
    method_values = df_result['Validation method'] == method
    df_method = df_result[method_values]
    df_method.sort_values(metric,ascending=False, inplace=True)

    ax = sns.barplot(x='Classifier', y=metric, data=df_method, ci=None)
    ax.bar_label(ax.containers[0])
    ax.set_title(f'{method}')

    # Show the plot
    plt.show()