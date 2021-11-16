# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys

import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineEDA:
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

def eda_baseline_boxplot(baseline):
    control = baseline.loc[baseline['class'] == 0]

    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline["mean"])
    plt.show()

    #plt.show()
    print(control.describe())
    print(control)



if __name__ == '__main__':

    baseline = sys.stdin
    eda_baseline_boxplot(baseline)