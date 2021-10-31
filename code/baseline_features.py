import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns


#Code to investigate the values from the file schizophrenia-features.csv


def loadFileCSV(dir):
    data = pd.read_csv(dir, delimiter=',')

    return data


def create_structure():
    dir_baseline = "../psykose/schizophrenia-features.csv"

    content_baseline = loadFileCSV(dir_baseline)


    print(content_baseline)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

    return content_baseline

def stats_baseline_boxplot(baseline):
    control = baseline.loc[baseline['class'] == 0]

    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline["f.mean"])
    plt.show()

    #plt.show()
    print(control.describe())
    print(control)


if __name__ == '__main__':
    baseline = create_structure()
    stats_baseline_boxplot(baseline)

