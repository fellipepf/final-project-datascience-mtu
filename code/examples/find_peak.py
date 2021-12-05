import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py
import utils
from psykose import LoadDataset

#from code.psykose import LoadDataset


def get_serie():
    control, patient = LoadDataset().get_dataset_by_class()
    print("")
    id = "control_5"
    df_serie = control[id]["timeserie"]

    df_serie["datetime"] = pd.to_datetime(df_serie["timestamp"])

    values = df_serie["activity"].values
    return values


def get_serie_2():
    y = np.array([1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,
       1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,
       2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1])
    return y

def example_values():
    x = electrocardiogram()[2000:4000]
    plt.plot(x)
    plt.show()

    return x

def find_peak(x):
    peaks, _ = find_peaks(x, height=3)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()

def find_peak_above_avg(x):
    avg = x.mean()

    peaks, _ = find_peaks(x, height=avg)
    print(f"AVG: {avg} Peaks: {len(peaks)}")
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()

#get_serie()

if __name__ == '__main__':
    serie = get_serie()[:500]
    plt.plot(serie)
    plt.show()

    find_peak_above_avg(serie)