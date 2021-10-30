#import matplotlib
import numpy as np
import pandas as pd
import glob
import re

from enum import Enum
import matplotlib.pyplot as plt

class Target(Enum):
    PATIENT = 1
    CONTROL = 0

def loadFileCSV(dir, Target):
    content = {}
    files = glob.glob(dir)
    for file in files:
        data = pd.read_csv(file, delimiter=',' )
        key = getName(file)
        content[key] = {}
        content[key]['timeserie'] = data
        content[key]['target'] = Target


    return content



def loadFiles(dir):

    content = list()
    files = glob.glob(dir)
    for name in files:
        with open(name, encoding="utf8", errors='ignore' ) as f:
            values = f.read()
            values = values.splitlines()
            content.append(values)
    return content


def getName(name):
    x = name.split("/")
    x = x[3].split(".")

    return x[0]


def create_structure():
    dir_control = "../psykose/control/*.csv"
    dir_patient = "../psykose/patient/*.csv"

    contentControl = loadFileCSV(dir_control, Target.CONTROL)
    contentPatient = loadFileCSV(dir_patient, Target.PATIENT)


    #print(contentControl)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

    return contentControl, contentPatient


def statistics(control):
    df_stats = pd.DataFrame()

    for key in set(control.keys()).difference({'target'}):
        value = control[key]

        df = value['timeserie']

        desc = df['activity'].describe()

        s = pd.Series(key, index=['userid'])
        desc = s.append(desc )

        df_stats = df_stats.append(desc, ignore_index=True  )

    return df_stats


def export_df_to_html(df, name = None):
    html = df.to_html()

    if name is None:
        name = "df.html"
    else:
        name = name + ".html"

    # write html to file
    text_file = open(name, "w")
    text_file.write(html)
    text_file.close()



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    control, patient = create_structure()

    print(control)

    #statsControl = statistics(control)
    #statsPatient = statistics(patient)

    #export_df_to_html(statsControl, "statsControl")
    #export_df_to_html(statsPatient, "statsPatient")

    # graph

    userid = "patient_1"
    df_patient = patient[userid]['timeserie']

    df_patient.plot(x="timestamp",y="activity")
    plt.show()


    # the average value for each hour of the day
    #the activity increase after 6am and starts to decrease after 8pm
    df_patient["datetime"] = pd.to_datetime(df_patient["timestamp"])
    fig, axs = plt.subplots(figsize=(12, 4))
    df_patient.groupby(df_patient["datetime"].dt.hour)["activity"].mean().plot(
        kind='line', rot=0, ax=axs)
    plt.xlabel("Hour of the day");  # custom x label using matplotlib
    plt.ylabel("activity");
    plt.show()

    userid = "control_1"
    df_control = control[userid]['timeserie']

    # the average value for each hour of the day
    #the activity increase after 6am and starts to decrease after 8pm
    #for control person the behaviour of the activity looks like the same
    df_control["datetime"] = pd.to_datetime(df_control["timestamp"])
    fig, axs = plt.subplots(figsize=(12, 4))
    df_control.groupby(df_control["datetime"].dt.hour)["activity"].mean().plot(
        kind='line', rot=0, ax=axs)
    plt.xlabel("Hour of the day");  # custom x label using matplotlib
    plt.ylabel("activity");
    plt.show()