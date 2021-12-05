import matplotlib.pyplot as plt
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

def fast_fourier_transform(serie):
    pd.Series(np.fft.fft(pd.Series(serie))).plot()
    plt.show()


if __name__ == '__main__':
    serie = get_serie()[:500]
    plt.plot(serie)
    plt.show()

    fast_fourier_transform(serie)
