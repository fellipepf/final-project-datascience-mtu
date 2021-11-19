

# ------------------------------------------
# IMPORTS
# ------------------------------------------
import glob
from enum import Enum
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import baseline_eda
from collected_days import DaysCollected
import utils

class Target(Enum):
    PATIENT = 1
    CONTROL = 0

class LoadDataset:

    def __init__(self):
        self.control, self.patient = self.create_structure()

    def loadFileCSV(self, dir, Target):
        content = {}
        files = glob.glob(dir)
        for file in files:
            data = pd.read_csv(file, delimiter=',' )
            key = self.getName(file)
            content[key] = {}
            content[key]['timeserie'] = data
            content[key]['target'] = Target

        return content

    def getName(self, name):
        return os.path.basename(name).split('.')[0]


    def create_structure(self):
        #dir_control = "../psykose/control/*.csv"
        dir_control = "/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/psykose/control/*.csv"
        #dir_patient = "../psykose/patient/*.csv"
        dir_patient = "/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/psykose/patient/*.csv"

        contentControl = self.loadFileCSV(dir_control, Target.CONTROL)
        contentPatient = self.loadFileCSV(dir_patient, Target.PATIENT)

    #print(contentControl)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

        return contentControl, contentPatient

    def get_dataset(self):
        return self.control, self.patient


class PreProcessing:
    def __init__(self):
        pass



'''
This function reduce the dataset to the number of days for each person
'''
def process_dataset_byday(dataset):
    dic_result = dict()

    days_collected = DaysCollected()
    days = days_collected.get_days_collected()

    for key in set(dataset.keys()).difference({'target'}):
        value = dataset[key]

        df = value['timeserie']
        user_class = value['target']

        df["datetime"] = pd.to_datetime(df["timestamp"])
        #group_day = df.groupby(df["datetime"].dt.day)['activity']

        n_days_limit = days.loc[days['id'] == key]['days'].item()
        group_day = df.groupby(pd.Grouper(key='datetime', freq='D'))
        group_n_days = list(group_day)[:n_days_limit]

        # get the second element from tuple in a list using comprehension list
        df_days = [tuple[1] for tuple in group_n_days]

        #df_result.extend(group_n_days)

        #transform list of dataframes to only one dataframe
        #df_all = pd.concat(df_days)
        #df_all['class'] = user_class

        dic_result[key] = {}
        dic_result[key]['timeserie'] = df_days
        dic_result[key]['user_class'] = user_class

    return dic_result

'''
    structure
     userid - class - mean - sd - prop_zeros   -> day 1
     userid - class - mean - sd - prop_zeros   -> day 2
'''
def generate_baseline(dataset):
    df_stats = pd.DataFrame()

    col = DaysCollected()
    days = col.get_days_collected()

    for key in set(dataset.keys()).difference({'target'}):
        value = dataset[key]

        df = value['timeserie']
        user_class = value['target']

        df["datetime"] = pd.to_datetime(df["timestamp"])
        #group_day = df.groupby(df["datetime"].dt.day)['activity']

        n_days_limit = days.loc[days['id'] == key]['days'].item()
        group_day = df.groupby(pd.Grouper(key='datetime', freq='D'))
        group_n_days = list(group_day)[:n_days_limit]

        for daily_serie in group_n_days:
            date = daily_serie[0]

            mean = daily_serie[1]['activity'].mean()
            sd = np.std(daily_serie[1]['activity'])

            count_zero = (daily_serie[1]['activity']==0).sum()
            daily_serie_size = daily_serie[1]['activity'].size
            proportion_zero = count_zero/daily_serie_size

            kurtosis_value = kurtosis(daily_serie[1]['activity'], fisher=False)
            skewness = skew(daily_serie[1]['activity'])

            row_day = {'userid': key, 'class': user_class.value, 'date':date, 'mean': mean, 'sd': sd, 'prop_zero': proportion_zero, 'kurtosis': kurtosis_value, 'skew': skewness}
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




#
def graph_timeserie(patient):
    # graph

    userid = 'patient_1'
    df_patient = patient[userid]['timeserie']

    df_patient.plot(x="timestamp",y="activity")
    plt.show()


def graph_patient_avg_by_hour(patient):
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

def graph_control_avg_by_hour(control):

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

# https://scholarspace.manoa.hawaii.edu/bitstream/10125/64135/0317.pdf
def graph_activity_by_frequency(time_serie):
    #define activity ranges of each state below
    deep_sleep = 0
    sleep = ""
    aw = ""


def new_features(dataset):
   # list_people = [dataset[key]['timeserie'] for key in dataset]
    df_day_night = {}

    for key in set(dataset.keys()):
        value = dataset[key]

        list_tm = value['timeserie']
        user_class = value['user_class']
        df_day_night[key] = {}

        list_day = list()
        list_night = list()
        for dftm in list_tm:
            dftm.index = dftm['datetime']

            day = dftm.between_time('6:00', '17:59')
            list_day.append(day)

            #this is one day range so after 23:59 is not in this period of time
            night = dftm.between_time('18:00', '23:59')
            list_night.append(night)

        df_day_night[key]['day'] = list_day
        df_day_night[key]['night'] = list_night
        df_day_night[key]['user_class'] = user_class

    return df_day_night

'''

This function calculates the mean, SD and proportion of zeros
input:
output:
'''
def calculate_statistics(daily_serie):
    mean = daily_serie['activity'].mean()
    sd = np.std(daily_serie['activity'])

    count_zero = (daily_serie['activity'] == 0).sum()
    daily_serie_size = daily_serie['activity'].size
    proportion_zero = count_zero / daily_serie_size

    kurtosis_value = kurtosis(daily_serie['activity'], fisher=False)
    skewness = skew(daily_serie['activity'])

    return mean, sd, proportion_zero, kurtosis_value, skewness

def stats_day_night(df_day_night):
    for key in set(df_day_night.keys()):
        value = df_day_night[key]
        user_class = value['user_class']

        list_day = value['day']
        df_stats = pd.DataFrame()
        for daily_serie in list_day:
            mean, sd, proportion_zero, kurtosis, skewness = calculate_statistics(daily_serie)
#, 'class': user_class.value, 'date':date,
            row_stats = {'userid': key, 'class': user_class.value, 'mean': mean, 'sd': sd, 'prop_zero': proportion_zero, 'kurtosis': kurtosis, 'skew': skewness}
            df_stats = df_stats.append(row_stats, ignore_index=True )
        df_day_night[key]['day_stats'] = df_stats

        list_night = value['night']
        df_stats = pd.DataFrame()
        for daily_serie in list_night:
            mean, sd, proportion_zero, kurtosis, skewness = calculate_statistics(daily_serie)

            row_stats = {'userid': key, 'class': user_class.value, 'mean': mean, 'sd': sd, 'prop_zero': proportion_zero, 'kurtosis': kurtosis, 'skew': skewness}
            df_stats = df_stats.append(row_stats, ignore_index=True )
        df_day_night[key]['night_stats'] = df_stats

    return df_day_night

def eda_day_night(day_night):
    pass

if __name__ == '__main__':

    ##configuration
    export_baseline_to_html = False

    # EDA
    show_timeseries_graph = False
    show_baseline_boxplot = True

    control, patient = LoadDataset().get_dataset()

    #print(control)

    # EDA
    #graph of entire timeserie of a given person
    if show_timeseries_graph:
        graph_timeserie(patient)
        graph_patient_avg_by_hour(patient)
        graph_control_avg_by_hour(control)


    # dataset with day and night
    processed_dataset = process_dataset_byday(control)
    df_day_night = new_features(processed_dataset)
    df_day_night = stats_day_night(df_day_night)

    # graphs for day night


    ####

    # dataset
    baselineControl = generate_baseline(control)
    baselinePatient = generate_baseline(patient)


    if export_baseline_to_html:
        beda = baseline_eda.BaselineEDA()
        beda.export_df_to_html(baselineControl, "statsControl")
        beda.export_df_to_html(baselinePatient, "statsPatient")
        #beda.export_df_to_html(df_day_night, "stats")

    #baseline generated from the time series

    if show_baseline_boxplot:
        join_baselines = pd.concat([baselineControl, baselinePatient])
        #baseline_eda.eda_baseline_boxplot_mean(join_baselines)
        #baseline_eda.eda_baseline_boxplot(join_baselines, utils.Feature.SKEW)
        baseline_eda.eda_baseline_boxplot_remove_outliers(join_baselines, utils.Feature.SKEW, True)

    graph_activity_by_period(control)


