# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys

import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns




def stats_baseline_boxplot(baseline):
    control = baseline.loc[baseline['class'] == 0]

    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline["mean"])
    plt.show()

    #plt.show()
    print(control.describe())
    print(control)



if __name__ == '__main__':

    baseline = sys.stdin
    stats_baseline_boxplot(baseline)