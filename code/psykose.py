


# ------------------------------------------
# IMPORTS
# ------------------------------------------
import numpy
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
    #structure
    # userid - class - mean - sd - prop_zeros   -> day 1
    # userid - class - mean - sd - prop_zeros   -> day 2

    df_stats = pd.DataFrame()

    for key in set(control.keys()).difference({'target'}):
        value = control[key]

        df = value['timeserie']
        user_class = value['target']

        df["datetime"] = pd.to_datetime(df["timestamp"])
        group_day = df.groupby(df["datetime"].dt.day)['activity']

        for daily_serie in group_day:
            mean = daily_serie[1].mean()
            sd = numpy.std(daily_serie[1])

            count_zero = np.where(daily_serie[1]==0)[0].size
            daily_serie_size = daily_serie[1].size
            proportion_zero = count_zero/daily_serie_size

            row_day = {'userid': key, 'class': user_class.value, 'mean': mean, 'sd': sd, 'prop_zero': proportion_zero}
            df_stats = df_stats.append(row_day, ignore_index=True )

            print(f'{key} mean {mean} sd {sd} zeros {count_zero}, total {daily_serie_size} -> {proportion_zero}')



        # countzeros = df.groupby(df["datetime"].dt.day)['activity'].apply(lambda x: (x == 0).sum()).reset_index(name='count')
        # total_day = df.groupby(df["datetime"].dt.day)['activity'].count()
        # print(key)
        # for cz, total in zip(countzeros['count'].values, total_day):
        #     print(f'zeros {cz}, total {total} -> {cz/total}')
        #     proportion_zeros = cz/total
        #




        # desc = df['activity'].describe()
        #
        # s = pd.Series(key, index=['userid'])
        # desc = s.append(desc )
        #
        # df_stats = df_stats.append(desc, ignore_index=True  )

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

#
def graph_timeserie(patient):
    # graph

    userid = "patient_1"
    df_patient = patient[userid]['timeserie']

    df_patient.plot(x="timestamp",y="activity")
    plt.show()


def graph_patient_avg_by_hour():
    userid = "patient_1"
    df_patient = patient[userid]['timeserie']


    # the average value for each hour of the day
    #the activity increase after 6am and starts to decrease after 8pm
    df_patient["datetime"] = pd.to_datetime(df_patient["timestamp"])
    fig, axs = plt.subplots(figsize=(12, 4))
    df_patient.groupby(df_patient["datetime"].dt.hour)["activity"].mean().plot(
        kind='line', rot=0, ax=axs)
    plt.xlabel("Hour of the day");  # custom x label using matplotlib
    plt.ylabel("activity");
    plt.show()

def graph_control_avg_by_hour():

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

def graph_activity_by_period(control):
    userid = "control_5"

    df_control = control[userid]['timeserie']
    df_control['datetime'] = pd.to_datetime(df_control['timestamp'])
    df_control = df_control.set_index('datetime')

    morning = df_control.between_time('6:00', '11:59')
    afternoon = df_control.between_time('12:00', '17:59')
    night = df_control.between_time('18:00', '23:59')

    #TODO define the periods of time
    #morning
    morning.plot(x="timestamp",y="activity")
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.show()


def graph_activity_by_frequency():
    #define activity ranges of each state below
    deep_sleep = ""
    sleep = ""
    aw = ""




if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    control, patient = create_structure()

    print(control)

    #graph of entire timeserie of a given person
    #graph_timeserie(patient)

    statsControl = statistics(control)
    statsPatient = statistics(patient)

    export_df_to_html(statsControl, "statsControl")
    export_df_to_html(statsPatient, "statsPatient")

    graph_activity_by_period(control)


