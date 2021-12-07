import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
sys.path.insert(0,"/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/code/")  # path contains python_file.py
import utils
from psykose import LoadDataset, PreProcessing
from sklearn.utils import shuffle



def get_serie():
    control_patient = LoadDataset().get_dataset_joined()
    df_dataset = PreProcessing(control_patient).df_dataset

    #shuffle df
    df_dataset = shuffle(df_dataset)

    classes = [c.name for c in df_dataset['class']]

    return classes

def enconding(serie):
    encoded = pd.get_dummies(serie)

    print(encoded.shape)
    print(f"after encoding: {encoded[:20]}")


if __name__ == '__main__':
    serie = get_serie()[:500]

    print(f"Before encoding: {serie}")
    plt.plot(serie)
    plt.show()

    enconding(serie)

