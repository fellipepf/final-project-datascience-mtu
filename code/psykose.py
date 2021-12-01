

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
from collections import OrderedDict

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

    def get_dataset_by_class(self):
        return self.control, self.patient

    def get_dataset_joined(self):
        return control | patient


class PreProcessing:
    def __init__(self):
        pass



'''
This function reduce the dataset to the number of days for each person
'''
def process_dataset_byday(dataset):
    dic_result = dict()
    remove_fist_day = True

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

        if remove_fist_day:
            list_days = list(group_day)

            #remove first day and
            # slice n number of elements defined
            group_n_days = list_days[1:n_days_limit +1]
        else:
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
Generate a paper baseline
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

'''
New features
'''

def split_time_period(dataset):
   # list_people = [dataset[key]['timeserie'] for key in dataset]
    df_day_night = {}

    for key in set(dataset.keys()):
        value = dataset[key]
        # list with the serie by day
        list_tm = value['timeserie']
        user_class = value['user_class']
        df_day_night[key] = {}

        list_morning = list()
        list_afternoon = list()
        list_evening = list()

        for dftm in list_tm:
            dftm.index = dftm['datetime']

            #from midnight to midday
            morning = dftm.between_time('0:00', '6:59')
            list_morning.append(morning)

            #this is one day range so after 23:59 is not in this period of time
            afternoon = dftm.between_time('7:00', '19:00', include_start = True, include_end = False)
            list_afternoon.append(afternoon)

            evening = dftm.between_time('19:00', '00:00', include_start = True, include_end = False)
            list_evening.append(evening)

        df_day_night[key]['morning'] = list_morning
        df_day_night[key]['afternoon'] = list_afternoon
        df_day_night[key]['evening'] = list_evening
        df_day_night[key]['user_class'] = user_class

    return df_day_night

def build_baseline_from_timeperiod(df_day_night):
    df_stats = pd.DataFrame()

    for key in set(df_day_night.keys()):
        value = df_day_night[key]

        morning_df_list = value['morning']
        afternoon_df_list = value['afternoon']
        evening_df_list = value['evening']

        user_class = value['user_class']
        for morning, afternoon, evening in zip(morning_df_list, afternoon_df_list, evening_df_list):
            morning_mean, morning_sd, morning_prop_zero, morning_kurtosis, morning_skewness = calculate_statistics(morning)
            afternoon_mean, afternoon_sd, afternoon_prop_zero, afternoon_kurtosis, afternoon_skewness = calculate_statistics(afternoon)
            evening_mean, evening_sd, evening_prop_zero, evening_kurtosis, evening_skewness = calculate_statistics(evening)

            row_stats = {'userid': key, 'class': user_class.value,
                         'morning_mean': morning_mean, 'morning_sd': morning_sd, 'morning_prop_zero': morning_prop_zero, 'morning_kurtosis': morning_kurtosis, 'morning_skewness': morning_skewness,
                         'afternoon_mean': afternoon_mean, 'afternoon_sd': afternoon_sd, 'afternoon_prop_zero': afternoon_prop_zero, 'afternoon_kurtosis': afternoon_kurtosis, 'afternoon_skewness': afternoon_skewness,
                         'evening_mean': evening_mean, 'evening_sd': evening_sd, 'evening_prop_zero': evening_prop_zero, 'evening_kurtosis': evening_kurtosis, 'evening_skewness': evening_skewness

                         }
            df_stats = df_stats.append(row_stats, ignore_index=True)

    df_stats = df_stats[['userid', 'class', 'morning_mean', 'morning_sd', 'morning_prop_zero', 'morning_kurtosis', 'morning_skewness',
                         'afternoon_mean', 'afternoon_sd', 'afternoon_prop_zero', 'afternoon_kurtosis', 'afternoon_skewness',
                         'evening_mean', 'evening_sd', 'evening_prop_zero', 'evening_kurtosis', 'evening_skewness'
                         ]]

    return df_stats

        #df["datetime"] = pd.to_datetime(df["timestamp"])
        # group_day = df.groupby(df["datetime"].dt.day)['activity']



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






def eda_baseline_date_range(baseline):
    df_stats = pd.DataFrame()
    for index, (key, value) in enumerate(baseline.items()):

        df_timeserie = value["timeserie"]
        user_class = value ["target"]

        df_timeserie['datetime'] = pd.to_datetime(df_timeserie['timestamp'])

        initial_date = df_timeserie.iloc[0]["datetime"]
        final_date = df_timeserie.iloc[-1]["datetime"]
        diff = final_date.date() - initial_date.date()
        total_days = diff.days +1

        #diferenca em segundo entre timestamp
        size = df_timeserie.shape[0]

        group_day = df_timeserie.groupby(pd.Grouper(key='datetime', freq='D'))
        list_days = list(group_day)
        days_first_day =  list_days[0][1].shape[0]
        days_second_day = list_days[1][1].shape[0]
        days_last_day = list_days[-1][1].shape[0]

        row_stats = {'userid': key, 'class': user_class.name,
                     "initial_date": initial_date,
                     "final_date": final_date,
                     "total_days": total_days,
                     "total_values": size,
                     "values_first_day": days_first_day,
                     "values_second_day": days_second_day, #represents middle of day
                     "values_last_day": days_last_day}
        df_stats = df_stats.append(row_stats, ignore_index=True )

    df_stats = df_stats[['userid', 'class', 'initial_date', 'final_date', 'total_days', 'total_values', 'values_first_day','values_second_day', 'values_last_day'  ]]

    return df_stats


if __name__ == '__main__':

    ##configuration
    export_baseline_to_html = False
    export_baseline_to_csv = True

    # EDA
    show_timeseries_graph = False
    show_baseline_boxplot = False
    generate_baseline_info = False

    # load dataset from csv files
    loadDataset = LoadDataset()
    control, patient = loadDataset.get_dataset_by_class()
    #print(control)
    control_patient = loadDataset.get_dataset_joined()


    # EDA
    if generate_baseline_info:
        range_info = eda_baseline_date_range(control)

    #graph of entire timeserie of a given person
    if show_timeseries_graph:
        graph_timeserie(patient)
        graph_patient_avg_by_hour(patient)
        graph_control_avg_by_hour(control)
        graph_activity_by_period(control)


    # dataset with day and night

    processed_dataset = process_dataset_byday(control_patient)
    df_time_period = split_time_period(processed_dataset)
    baseline_time_period = build_baseline_from_timeperiod(df_time_period)
    #df_day_night = stats_day_night(df_day_night)

    expBaseline = baseline_eda.ExportBaseline()
    expBaseline.generate_baseline_csv(baseline_time_period, "baseline_time_period.csv")


    # graphs for day night


    ####

    # Baseline
    # create paper baseline
    baselineControl = generate_baseline(control)
    baselinePatient = generate_baseline(patient)

    baseline_eda.check_nullvalues(baselinePatient)
    #eda_baseline_date_range(baselineControl)

    if export_baseline_to_html:
        beda = baseline_eda.ExportBaseline()
        beda.export_df_to_html(baselineControl, "statsControl")
        beda.export_df_to_html(baselinePatient, "statsPatient")
        #beda.export_df_to_html(df_day_night, "stats")

    #baseline generated from the time series
    join_baselines = pd.concat([baselineControl, baselinePatient])

    if show_baseline_boxplot:
        #baseline_eda.eda_baseline_boxplot_mean(join_baselines)
        #baseline_eda.eda_baseline_boxplot(join_baselines, utils.Feature.SKEW)
        baseline_eda.eda_baseline_boxplot_remove_outliers(join_baselines, utils.Feature.SKEW, True)

    if export_baseline_to_csv:
        beda = baseline_eda.ExportBaseline()
        beda.generate_baseline_csv(join_baselines)




