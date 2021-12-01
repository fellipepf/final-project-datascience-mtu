# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns


class ExportBaseline:
    def __init__(self):
        pass

    def export_df_to_html(self, df, name=None):
        html = df.to_html()

        if name is None:
            name = "df.html"
        else:
            name = name + ".html"

        # write html to file
        text_file = open(name, "w")
        text_file.write(html)
        text_file.close()

    # export baseline to csv
    def generate_baseline_csv(self, baselines, file_name=None):

        baselines["class_str"] = "c_0_1"
        try:
            baselines = baselines.drop(["date"], axis=1)
        except:
            print("Column not found")

        if file_name == None:
            file_name = 'my_baseline.csv'

        baselines.to_csv(file_name, index=False, header=True, line_terminator='\n', encoding='utf-8', sep=',')


def eda_baseline_boxplot_mean(baseline):
    control = baseline.loc[baseline['class'] == 0]

    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline["mean"])
    plt.show()

    #plt.show()
    print(control.describe())
    print(control)

def eda_baseline_boxplot(baseline, y_feature):
    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline[y_feature])
    plt.show()

def eda_baseline_boxplot_remove_outliers(baseline, y_feature, remove_outlier=False):
    baseline_witout_outlier = pd.DataFrame()

    if remove_outlier:
        baseline_witout_outlier = remove_outliers(baseline, y_feature)

    eda_baseline_boxplot(baseline_witout_outlier, y_feature)

def remove_outliers(baseline, y_feature):

    q25 = baseline[y_feature].quantile(0.25)
    q75 = baseline[y_feature].quantile(0.75)
    intr_qr = q75 - q25

    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, intr_qr))

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    print('Lower: %.3f, Upper: %.3f,' % (min, max))
    baseline[baseline[y_feature] > max] = np.nan
    baseline[baseline[y_feature] < min] = np.nan

    # remove rows with NAN based on column feature
    baseline = baseline.dropna(subset=[y_feature])

    return baseline

def check_nullvalues(baseline):
    #check for null values
    print(baseline.isnull().sum())


if __name__ == '__main__':

    baseline = sys.stdin
    eda_baseline_boxplot_mean(baseline)