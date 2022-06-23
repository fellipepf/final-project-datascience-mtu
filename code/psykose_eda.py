
from psykose_dataset import LoadDataset, PreProcessing
import sys

import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt, dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
import dataframe_image as dfi
import baseline_eda
from enum import Enum

class Target(Enum):
    PATIENT = 1
    CONTROL = 0

    def __eq__(self, b):
        return self.value == b.value


def boxplot():
    pass

def lineplot():
    pass

def perform_eda_of_person(id):
    pass


def raw_dataset_get_series_from_user(dict_time_period, id):
    output = pd.DataFrame()

    if type(dict_time_period) is dict:
        if id in dict_time_period:
            output = dict_time_period[id]

    output = output['timeserie']
    return output


def processed_dataset_get_series_from_user(processed_ds, id):
    if isinstance(processed_ds, pd.DataFrame):
        return processed_ds.loc[processed_ds['user'] == id]


def plot_line_graph(dict_time_period, id):

    df_series = pd.DataFrame()

    if type(dict_time_period) is dict:
        df_series = raw_dataset_get_series_from_user(dict_time_period, id)
    else:
        df_series = processed_dataset_get_series_from_user(dict_time_period, id)


    #list_df_morning = values['morning']
    #merged = pd.concat(list_df_morning)

    df_series["datetime"] = pd.to_datetime(df_series["timestamp"])
    timeserie = df_series.set_index('datetime')

    #plt.set_loglevel('WARNING')
    sns.set(rc={'figure.figsize': (11, 4)})
    sns.set_style("ticks")
    plot = sns.lineplot(x="datetime", y="activity", data=timeserie)
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
    #plot.axvline("%d-%m-%Y 06:00", color="red", linestyle="dashed")
    plt.title(f"Time Serie of user: {id}",fontsize= 20)
    plt.ylabel("Activity")
    plt.xticks(rotation=25)
    plt.show()


def eda_baseline_date_range(baseline):
    df_stats = pd.DataFrame()
    for index, (key, value) in enumerate(baseline.items()):

        df_timeserie = value["timeserie"]

        if "target" in value:
            user_class = value ["target"]
        elif "user_class" in value:
            user_class = value ["user_class"]

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


def graph_patient_avg_by_hour(patient):
    userid = "patient_1"
    df_patient = patient[userid]['timeserie']


    # the average value for each hour of the day
    #the activity increase after 6am and starts to decrease after 8pm
    df_patient["datetime"] = pd.to_datetime(df_patient["timestamp"])
    fig, axs = plt.subplots(figsize=(12, 4))
    df_patient.groupby(df_patient["datetime"].dt.hour)["activity"].mean().plot(
        kind='line', rot=0, ax=axs)
    plt.title(f"Average of activity by hour - {userid} ", fontsize=20)
    plt.xlabel("Hour of the day");  # custom x label using matplotlib
    plt.ylabel("activity")
    plt.grid()
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
    plt.title(f"Average of activity by hour - {userid} ")
    plt.xlabel("Hour of the day")  # custom x label using matplotlib
    plt.ylabel("activity")
    plt.grid()
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


def caculate_mean_hour_all_participant(mean_hour):
    #print(tabulate(mean_hour, headers='keys', tablefmt='psql'))
    transpose = [e.T for e in mean_hour]
    transpose = pd.DataFrame(transpose)
    df_mean_column = transpose.aggregate(['mean'])

    #back to one column shape
    df_mean_column = df_mean_column.T
    return df_mean_column

def graph_avg_activity_all_participants(dataset):
    mean_per_user = pd.DataFrame()

    for key in set(dataset.keys()).difference({'target'}):
        value = dataset[key]

        df = value['timeserie']
        user_class = value['target']

        mean_values = df.groupby(df["datetime"].dt.hour)["activity"].mean()
        row = {}
        row['target'] = user_class
        row['mean_hour'] = mean_values

        mean_per_user = mean_per_user.append(row, ignore_index=True)

    #control
    filter = mean_per_user["target"] == Target.CONTROL
    control_data = mean_per_user[filter]

    filter = mean_per_user["target"] == Target.PATIENT
    patient_data = mean_per_user[filter]

    mean_hour_control = control_data['mean_hour']
    mean_by_day_control = caculate_mean_hour_all_participant(mean_hour_control)
    mean_by_day_control = mean_by_day_control.reindex()

    mean_hour_patient = patient_data['mean_hour']
    mean_by_day_patient = caculate_mean_hour_all_participant(mean_hour_patient)
    mean_by_day_patient = mean_by_day_patient.reindex()

    df_to_plot = pd.DataFrame()
    df_to_plot['control'] = mean_by_day_control['mean'].tolist()
    df_to_plot['patient'] = mean_by_day_patient['mean'].tolist()


    #plt.figure(figsize=(20, 12))
    ax = sns.lineplot( data=df_to_plot, dashes = False)

    # add vertical line
    ax.vlines(x=[7], ymin=10, ymax=340, color='g', ls='--',label='test lines')
    ax.vlines(x=[19], ymin=10, ymax=340, color='g',ls='--', label='test lines')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.xlabel("Hour",fontsize= 15)
    plt.ylabel("Mean Activity",fontsize= 15)
    plt.title("Mean Activity by Hour of Patients and Control",fontsize= 20)
    plt.grid()
    plt.show()




if __name__ == '__main__':

    id_person = "control_12"

    generate_baseline_info = False
    show_timeseries_graph = True


    #load
    loadDataset = LoadDataset()
    control, patient = loadDataset.get_dataset_by_class()
    control_patient = loadDataset.get_dataset_joined()

    #plot_line_graph(control_patient, id_person)


    #Pre-procesing phase

    data_process = PreProcessing(control, patient)
    ts_byday = data_process.control_patient_byday
    df_timeserie = data_process.create_one_df_struture(ts_byday)

    #plot_line_graph(df_timeserie, id_person)

    baseline_eda.find_peak_above_avg(data_process.control_patient_byday)
    baseline_eda.plot_graph_one_day(data_process.control_patient_byday)
    baseline_eda.plot_graph_periods_of_day(data_process.control_patient_byday)

    if generate_baseline_info:
        range_info = eda_baseline_date_range(control)
        range_info_processed = eda_baseline_date_range(data_process.control_patient_byday)

    #graph of entire timeserie of a given person
    if show_timeseries_graph:
        #full dataset
        baseline_eda.graph_timeserie_byid(control_patient, 'patient_1')
        #baseline_eda.graph_timeserie_byid(control_patient, 'control_7')

        #day limit dataset
        control_patient_byday = data_process.control_patient_byday
        #baseline_eda.graph_timeserie_byid(control_patient_byday, 'control_7')
        baseline_eda.graph_timeserie_byid(control_patient_byday, 'patient_1')

        graph_patient_avg_by_hour(patient)
        graph_control_avg_by_hour(control)
        graph_activity_by_period(control)
        graph_avg_activity_all_participants(control_patient)