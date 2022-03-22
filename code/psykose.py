

# ------------------------------------------
# IMPORTS
# ------------------------------------------
import glob
from enum import Enum
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import dataframe_image as dfi
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from collections import OrderedDict

import seaborn as sns
from tabulate import tabulate
import matplotlib.ticker as ticker
import baseline_eda
import log_configuration
from psykose_dataset import LoadDataset
from psykose_dataset import PreProcessing
from collected_days import DaysCollected
import utils

class Target(Enum):
    PATIENT = 1
    CONTROL = 0

    def __eq__(self, b):
        return self.value == b.value

class CreateBaseline:
    def __init__(self, *args):

        if len(args) == 1:
            pass

        if len(args) == 2:
            pass

    def baseline_paper(self):
        pass

class PlotGraphs:
    def __init__(self, *args):
        pass


# def process_dataset_byday(dataset):
#     '''
#     This function reduce the dataset to the number of days for each person
#     '''
#
#     dic_result = dict()
#     remove_fist_day = True
#
#     days_collected = DaysCollected()
#     days = days_collected.get_days_collected()
#
#     for key in set(dataset.keys()).difference({'target'}):
#         value = dataset[key]
#
#         df = value['timeserie']
#         user_class = value['target']
#
#         df["datetime"] = pd.to_datetime(df["timestamp"])
#         #group_day = df.groupby(df["datetime"].dt.day)['activity']
#
#         n_days_limit = days.loc[days['id'] == key]['days'].item()
#         group_day = df.groupby(pd.Grouper(key='datetime', freq='D'))
#
#         if remove_fist_day:
#             list_days = list(group_day)
#
#             #remove first day and
#             # slice n number of elements defined
#             group_n_days = list_days[1:n_days_limit +1]
#         else:
#             group_n_days = list(group_day)[:n_days_limit]
#
#         # get the second element from tuple in a list using comprehension list
#         df_days = [tuple[1] for tuple in group_n_days]
#
#         #df_result.extend(group_n_days)
#
#         #transform list of dataframes to only one dataframe
#         #df_all = pd.concat(df_days)
#         #df_all['class'] = user_class
#
#         dic_result[key] = {}
#         dic_result[key]['timeserie'] = df_days
#         dic_result[key]['user_class'] = user_class
#
#     return dic_result


def generate_baseline(dataset):
    '''
    Generate a paper baseline - reproduced dataset
        structure
         userid - class - mean - sd - prop_zeros   -> day 1
         userid - class - mean - sd - prop_zeros   -> day 2
    '''

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

            row_day = {'userid': key, 'class': user_class.value, 'date':date, 'mean': mean, 'sd': sd, 'prop_zero': proportion_zero,
                       #'kurtosis': kurtosis_value,
                       #'skew': skewness
                       }
            df_stats = df_stats.append(row_day, ignore_index=True )


            #print(f'{key} mean {mean} sd {sd} zeros {count_zero}, total {daily_serie_size} -> {proportion_zero}')



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

def graph_patient_avg_by_hour(patient):
    userid = "patient_1"
    df_patient = patient[userid]['timeserie']


    # the average value for each hour of the day
    #the activity increase after 6am and starts to decrease after 8pm
    df_patient["datetime"] = pd.to_datetime(df_patient["timestamp"])
    fig, axs = plt.subplots(figsize=(12, 4))
    df_patient.groupby(df_patient["datetime"].dt.hour)["activity"].mean().plot(
        kind='line', rot=0, ax=axs)
    plt.title(f"Average of activity by hour - {userid} ")
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
    ax = sns.lineplot( data=df_to_plot)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.xlabel("Hour")
    plt.ylabel("Mean Activity")
    plt.title("Mean Activity by Hour of Patients and Control")
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

# https://scholarspace.manoa.hawaii.edu/bitstream/10125/64135/0317.pdf
def graph_activity_by_frequency(time_serie):
    #define activity ranges of each state below
    deep_sleep = 0
    sleep = ""
    aw = ""

'''
New features
'''


def split_time_period(dataset, periods):
    # list_people = [dataset[key]['timeserie'] for key in dataset]
    df_day_night = {}
    dataset_byday = dataset.control_patient_byday

    list_periods = periods.list_periods

    #itetate participants
    for key in set(dataset_byday.keys()):
        value = dataset_byday[key]
        # list with the serie by day
        list_tm = value['timeserie']
        user_class = value['user_class']
        df_day_night[key] = {}

        list_morning = list()
        list_afternoon = list()
        list_evening = list()

        days_concatenated = pd.concat(list_tm)

        #iterate days
        #for dftm in list_tm:
        days_concatenated.index = days_concatenated['datetime']

        for index, row in list_periods.iterrows():
            period_name = row['period_name']
            time_start = row['time_start']
            time_finish = row['time_finish']

            # from midnight to midday
            period_values = days_concatenated.between_time(time_start, time_finish)

            period_values_by_day = period_values.groupby(pd.Grouper(key='datetime', freq='D'))
            list_days = list(period_values_by_day)
            list_df_days = [e[1] for e in list_days]
            #
            #         if remove_fist_day:
            #             list_days = list(group_day)

            # #from midnight to midday
            # morning = days_concatenated.between_time('0:00', '6:59')
            # list_morning.append(morning)
            #
            # #this is one day range so after 23:59 is not in this period of time
            # afternoon = days_concatenated.between_time('7:00', '19:00', include_start = True, include_end = False)
            # list_afternoon.append(afternoon)
            #
            # evening = days_concatenated.between_time('19:00', '00:00', include_start = True, include_end = False)
            # list_evening.append(evening)

            df_day_night[key][period_name] = list_df_days
        #df_day_night[key]['afternoon'] = list_afternoon
        #df_day_night[key]['evening'] = list_evening
        df_day_night[key]['user_class'] = user_class

    return df_day_night



def build_baseline_from_timeperiod(df_periods_participants, periods_obj):
    df_stats_result = pd.DataFrame()


    dict_stats_periods = dict()

    #iterate DF of participants
    for key in set(df_periods_participants.keys()):
        periods = df_periods_participants[key]



        #iterate periods - except the user class
        list_day_values_per_period = dict()
        for period in set(periods.keys()).difference({'user_class'}):
            list_day_values_per_period = periods[period]
            user_class = periods['user_class']

            stats_of_periods = list()
            for day_values in list_day_values_per_period:
                stats = calculate_statistics(day_values)
                stats_of_periods.append(stats)

            dict_stats_periods[period] = stats_of_periods

        #create output DF
        #iterate stats values per periods
        df_stats_user = pd.DataFrame()
        for period_key in set(dict_stats_periods.keys()):
            stat_period = dict_stats_periods[period_key]

            df_stats_period = pd.DataFrame()
            #iterate stats for each day
            for stat_day in stat_period:
                #vars(stat_period[0])
                stat_values = vars(stat_day)

                output_df_row = dict()
                #iterate statistical values
                for stat_value in stat_values:
                    value = stat_values[stat_value]
                    column_name = f"{period_key}_{stat_value}"
                    output_df_row[column_name] = value

                #concat columns
                output_df_row = pd.DataFrame([output_df_row])
                df_stats_period = pd.concat([
                    df_stats_period,
                    output_df_row
                ])  #, axis=1

            #concat rows
            df_stats_user = pd.concat([
                df_stats_user,
                df_stats_period
            ], axis=1)

        df_stats_user['userid'] = key
        df_stats_user['class'] = user_class.value

        if df_stats_result.empty:
            df_stats_result = df_stats_user
        else:
            df_stats_user = df_stats_user.reset_index(drop=True)
            df_stats_result = df_stats_result.reset_index(drop=True)

            df_stats_result = pd.concat([
                df_stats_user,
                df_stats_result
            ], axis=0, ignore_index=True)

    return df_stats_result







'''



        #using the periods object to get the values from df_day_night

        morning_df_list = periods['morning']
        afternoon_df_list = periods['afternoon']
        evening_df_list = periods['evening']

        user_class = periods['user_class']

        for morning, afternoon, evening in zip(morning_df_list, afternoon_df_list, evening_df_list):
            morning_mean, morning_sd, morning_prop_zero, morning_kurtosis, morning_skewness, morning_peaks, morning_max, morning_median, morning_mad = calculate_statistics(morning)
            afternoon_mean, afternoon_sd, afternoon_prop_zero, afternoon_kurtosis, afternoon_skewness, afternoon_peaks, afternoon_max, afternoon_median, afternoon_mad = calculate_statistics(afternoon)
            evening_mean, evening_sd, evening_prop_zero, evening_kurtosis, evening_skewness, evening_peaks, evening_max, evening_median, evening_mad = calculate_statistics(evening)

            row_stats = {'userid': key, 'class': user_class.value,
                         'morning_mean': morning_mean,
                         'morning_sd': morning_sd,
                         'morning_prop_zero': morning_prop_zero,
                         'morning_kurtosis': morning_kurtosis,
                         'morning_skewness': morning_skewness,
                         'morning_peaks': morning_peaks,
                         'morning_max': morning_max,
                         'morning_median': morning_median,
                         'morning_mad': morning_mad,

                         'afternoon_mean': afternoon_mean,
                         'afternoon_sd': afternoon_sd,
                         'afternoon_prop_zero': afternoon_prop_zero,
                         'afternoon_kurtosis': afternoon_kurtosis,
                         'afternoon_skewness': afternoon_skewness,
                         'afternoon_peaks': afternoon_peaks,
                         'afternoon_max': afternoon_max,
                         'afternoon_median': afternoon_median,
                         'afternoon_mad': afternoon_mad,

                         'evening_mean': evening_mean,
                         'evening_sd': evening_sd,
                         'evening_prop_zero': evening_prop_zero,
                         'evening_kurtosis': evening_kurtosis,
                         'evening_skewness': evening_skewness,
                         'evening_peaks': evening_peaks,
                         'evening_max': evening_max,
                         'evening_median': evening_median,
                         'evening_mad': evening_mad

                         }
            df_stats = df_stats.append(row_stats, ignore_index=True)

    df_stats = df_stats[['userid', 'class',
                         'morning_mean', 'morning_sd', 'morning_prop_zero', 'morning_kurtosis', 'morning_skewness', 'morning_peaks','morning_max','morning_median', 'morning_mad',
                         'afternoon_mean', 'afternoon_sd', 'afternoon_prop_zero', 'afternoon_kurtosis', 'afternoon_skewness', 'afternoon_peaks','afternoon_max','afternoon_median', 'afternoon_mad',
                         'evening_mean', 'evening_sd', 'evening_prop_zero', 'evening_kurtosis', 'evening_skewness','evening_peaks', 'evening_max','evening_median', 'evening_mad'
                         ]]

    return df_stats
'''
        #df["datetime"] = pd.to_datetime(df["timestamp"])
        # group_day = df.groupby(df["datetime"].dt.day)['activity']

def create_baseline_output():
    pass

def graph_peaks(serie_period_time):
    serie = serie_period_time["activity"].values
    avg = serie.mean()

    peaks, _ = find_peaks(serie, height=avg)
    print(f"AVG: {avg} Peaks: {len(peaks)}")
    plt.plot(serie)
    plt.plot(peaks, serie[peaks], "x")
    plt.plot(np.zeros_like(serie), "--", color="gray")
    plt.show()

def count_peaks_above_mean(serie, avg):
    peaks, _ = find_peaks(serie, height=avg)
    return len(peaks)

'''

This function calculates the mean, SD and proportion of zeros
input:
output:
'''
def calculate_statistics(daily_serie):
    stats_values = dict()

    mean = daily_serie['activity'].mean()
    sd = np.std(daily_serie['activity'])

    count_zero = (daily_serie['activity'] == 0).sum()
    daily_serie_size = daily_serie['activity'].size
    proportion_zero = count_zero / daily_serie_size

    kurtosis_value = kurtosis(daily_serie['activity'], fisher=False)
    skewness = skew(daily_serie['activity'])

    count_peaks = count_peaks_above_mean(daily_serie['activity'], mean)
    max = daily_serie['activity'].max()
    median = daily_serie['activity'].median()

    #median absolute deviation
    x = daily_serie['activity']  # pd.Series()
    mad = (x - x.median()).abs().median()

    stats = StatisticalFeatures()
    stats.add_stat_values(mean, sd, proportion_zero, kurtosis_value, skewness, count_peaks, max, median, mad )

    return stats
    #return mean, sd, proportion_zero, kurtosis_value, skewness, count_peaks, max, median, mad

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


class PeriodsOfDay:

    list_periods = pd.DataFrame()

    def add_period(self, period_name, time_start, time_finish):
        period = dict()
        period['period_name'] = period_name
        period['time_start'] = time_start
        period['time_finish'] = time_finish

        PeriodsOfDay.list_periods = PeriodsOfDay.list_periods.append(period, ignore_index=True)

class StatisticalFeatures:

    def add_stat_values(self, mean, sd, prop_zero, kurtosis, skewness, peaks, max, median, mad):
        self.mean = mean
        self.sd = sd
        self.prop_zero = prop_zero
        self.kurtosis = kurtosis
        self.skewness = skewness
        self.peaks = peaks
        self.max = max
        self.median = median
        self.mad = mad


if __name__ == '__main__':
    logger = log_configuration.logger

    ##configuration

    export_baseline_to_html = False
    export_reproduced_dataset_to_csv = False
    export_new_features_dataset_to_csv = True

    # EDA
    show_timeseries_graph = False
    show_baseline_boxplot = False
    generate_baseline_info = False

    # load dataset from csv files
    loadDataset = LoadDataset()
    control, patient = loadDataset.get_dataset_by_class()
    #print(control)
    control_patient = loadDataset.get_dataset_joined()

    data_process = PreProcessing(control, patient)

    baseline_eda.find_peak_above_avg(data_process.control_patient_byday)
    baseline_eda.plot_graph_one_day(data_process.control_patient_byday)
    baseline_eda.plot_graph_periods_of_day(data_process.control_patient_byday)


    # EDA
    if generate_baseline_info:
        range_info = eda_baseline_date_range(control)
        range_info_processed = eda_baseline_date_range(data_process.control_patient_byday)

    #graph of entire timeserie of a given person
    if show_timeseries_graph:
        #full dataset
        baseline_eda.graph_timeserie_byid(control_patient, 'patient_1')
        baseline_eda.graph_timeserie_byid(control_patient, 'control_7')

        #day limit dataset
        control_patient_byday = data_process.control_patient_byday
        baseline_eda.graph_timeserie_byid(control_patient_byday, 'control_7')
        baseline_eda.graph_timeserie_byid(control_patient_byday, 'patient_1')

        graph_patient_avg_by_hour(patient)
        graph_control_avg_by_hour(control)
        graph_activity_by_period(control)
        graph_avg_activity_all_participants(control_patient)


    # dataset in time periods
    periods = PeriodsOfDay()

    #original division of time period
    periods.add_period('morning', '0:00', '6:59')
    periods.add_period('afternoon', '7:00', '19:00')
    periods.add_period('evening', '19:00', '00:00')



    processed_dataset = PreProcessing(control_patient)
    df_time_period = split_time_period(processed_dataset, periods)
    baseline_time_period = build_baseline_from_timeperiod(df_time_period, periods)
    #df_day_night = stats_day_night(df_day_night)

    baseline_eda.baseline_dimension(baseline_time_period)
    baseline_eda.plot_line_graph(df_time_period)

    list_morning_values = ['morning_mean', 'morning_sd', 'morning_prop_zero',
            'morning_kurtosis', 'morning_skewness']
    baseline_eda.box_plot_simple(baseline_time_period, list_morning_values)

    if export_new_features_dataset_to_csv:
        expBaseline = baseline_eda.ExportBaseline()
        expBaseline.generate_baseline_csv(baseline_time_period, "baseline_time_period_new.csv")


    # graphs for day night


    ####

    # Baseline
    # create paper baseline - reproduced dataset
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

    if export_reproduced_dataset_to_csv:
        beda = baseline_eda.ExportBaseline()
        beda.generate_baseline_csv(join_baselines)

    export_baseline_to_table_image = True
    if export_baseline_to_table_image:
        # m = control.get("control_1")['timeserie']
        # m = m.head()
        # dfi.export(m, 'output_images/control.png')

        baseline_eda.create_df_table(join_baselines, "df_processed")
        #baseline_eda.create_df_table(baseline_time_period, "df_processed_time_period")




