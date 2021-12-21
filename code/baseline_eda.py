# ------------------------------------------
# IMPORTS
# ------------------------------------------
import sys

import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt, dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from scipy.signal import find_peaks



class ExportBaseline:
    def __init__(self):
        pass

    def export_df_to_html(self, df, name=None):
        html = df.to_html()

        if name is None:
            name = "df.html"
        else:
            name = name + ".html"

        # write html to file
        text_file = open(name, "w")
        text_file.write(html)
        text_file.close()

    # export baseline to csv
    def generate_baseline_csv(self, baselines, file_name=None):

        baselines["class_str"] = "c_0_1"
        try:
            baselines = baselines.drop(["date"], axis=1)
        except:
            print("Column not found")

        if file_name == None:
            file_name = 'my_baseline.csv'

        baselines.to_csv(file_name, index=False, header=True, line_terminator='\n', encoding='utf-8', sep=',')


def eda_baseline_boxplot_mean(baseline):
    control = baseline.loc[baseline['class'] == 0]

    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline["mean"])
    plt.show()

    #plt.show()
    print(control.describe())
    print(control)

def eda_baseline_boxplot(baseline, y_feature):
    sns.set_style("darkgrid")
    sns.boxplot(x=baseline["class"], y=baseline[y_feature])
    plt.show()

def eda_baseline_boxplot_remove_outliers(baseline, y_feature, remove_outlier=False):
    baseline_witout_outlier = pd.DataFrame()

    if remove_outlier:
        baseline_witout_outlier = remove_outliers(baseline, y_feature)

    eda_baseline_boxplot(baseline_witout_outlier, y_feature)

def remove_outliers(baseline, y_feature):

    q25 = baseline[y_feature].quantile(0.25)
    q75 = baseline[y_feature].quantile(0.75)
    intr_qr = q75 - q25

    #print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, intr_qr))

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    print('Lower: %.3f, Upper: %.3f,' % (min, max))
    baseline[baseline[y_feature] > max] = np.nan
    baseline[baseline[y_feature] < min] = np.nan

    # remove rows with NAN based on column feature
    baseline = baseline.dropna(subset=[y_feature])

    return baseline

def check_nullvalues(baseline):
    #check for null values
    print(baseline.isnull().sum())


#
def graph_timeserie(patient):
    # graph

    userid = 'patient_1'
    df_patient = patient[userid]['timeserie']

    df_patient.plot(x="timestamp",y="activity")
    plt.show()

def graph_timeserie_byid(df_dataset, user_id='patient_1'):

    if user_id == None:
        user_id = 'patient_1'
    df_dataset = df_dataset.set_index('datetime')
    df_patient = df_dataset[user_id]['timeserie']

    #df_patient.plot(x="timestamp",y="activity")

    sns.lineplot(x="timestamp", y="activity", data=df_patient)
    sns.set(style='dark', )
    plt.xlabel("x-axis")
    plt.title("title")
    plt.show()


##### Baseline period of the day EDA graphs
def baseline_dimension(baseline_time_period):
    shape = baseline_time_period.shape
    print(f'Lines: {shape[0]} - columns: {shape[1]}')

def box_plot_simple(baseline_time_period, list_columns):
    #baseline_eda.eda_baseline_boxplot_remove_outliers(baseline_time_period, list, True)
    baseline_time_period.boxplot(column=list_columns)

    plt.figure(figsize=(20, 6))
    plt.show()


def plot_line_graph(df_time_period):
    id = "control_31"
    values = df_time_period[id]

    list_df_morning = values['morning']
    merged = pd.concat(list_df_morning)

    sns.set(rc={'figure.figsize': (11, 4)})
    merged['activity'].plot(linewidth=0.5);
    plt.show()


def plot_graph_one_day(df_time_period, id_user=None):
    id_user = "control_31"
    id_user = "patient_1"
    values = df_time_period[id_user]

    df_first_day = values['timeserie'][0]


    sns.set(rc={'figure.figsize': (11, 4)})
    sns.set_style("ticks")
    plot = sns.lineplot(x="datetime", y="activity", data=df_first_day)
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
    #plot.axvline("%d-%m-%Y 06:00", color="red", linestyle="dashed")
    plt.title(f"Time Serie of user: {id_user}")
    plt.ylabel("Activity")
    plt.xticks(rotation=25)
    plt.show()



def plot_graph_periods_of_day(df_time_period, id_user=None):
    id_user = "patient_1"

    values = df_time_period[id_user]

    df_first_day = values['timeserie'][0]
    df_first_day = df_first_day.set_index('datetime')
    # from midnight to midday
    morning = df_first_day.between_time('0:00', '6:59')

    # this is one day range so after 23:59 is not in this period of time
    afternoon = df_first_day.between_time('7:00', '19:00', include_start=True, include_end=False)
    evening = df_first_day.between_time('19:00', '00:00', include_start=True, include_end=False)

    fig, ax = plt.subplots(3, 1, figsize=(11, 4))

    plot = sns.lineplot(x="datetime", y="activity",  data=morning, ax=ax[0])
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #plot.xaxis.set_major_locator(ticker.LinearLocator(10))
    #ax[0].tick_params(labelrotation=25)
    #ax[0].set_title('Morning')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Activity")



    plot = sns.lineplot(x="datetime", y="activity", data=afternoon, ax=ax[1])
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #ax[1].tick_params(labelrotation=25)
    #ax[1].set_title('Afternoon')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Activity")


    plot = sns.lineplot(x="datetime", y="activity",  data=evening, ax=ax[2])
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #ax[2].tick_params(labelrotation=25)
    #ax[2].set_title('Evening')
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Activity")


    fig.suptitle(f"Time Serie of user: {id_user}")

    plt.show()

def find_peak_above_avg(df_time_period, id_user=None):

    if id_user == None:
        id_user = "patient_1"

    values = df_time_period[id_user]

    df_first_day = values['timeserie'][0]
    df_first_day = df_first_day.set_index('datetime')

    fig, ax = plt.subplots(3, 1, figsize=(11, 4))

    morning = df_first_day.between_time('0:00', '6:59')
    avg_morning = morning['activity'].mean()
    morning_values = morning['activity'].values

    morning_peaks, _ = find_peaks(morning_values, height=avg_morning)
    ax[0].plot(morning_values)
    ax[0].plot(morning_peaks, morning_values[morning_peaks], "x")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Activity")

    # this is one day range so after 23:59 is not in this period of time
    afternoon = df_first_day.between_time('7:00', '19:00', include_start=True, include_end=False)
    avg_afternoon = afternoon['activity'].mean()
    afternoon_values = afternoon['activity'].values

    afternoon_peaks, _ = find_peaks(afternoon_values, height=avg_afternoon)
    ax[1].plot(afternoon_values)
    ax[1].plot(afternoon_peaks, afternoon_values[afternoon_peaks], "x")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Activity")


    evening = df_first_day.between_time('19:00', '00:00', include_start=True, include_end=False)
    avg_evening = evening['activity'].mean()
    evening_values = evening['activity'].values

    evening_peaks, _ = find_peaks(evening_values, height=avg_evening)
    ax[2].plot(evening_values)
    ax[2].plot(evening_peaks, evening_values[evening_peaks], "x")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Activity")



    fig.suptitle(f"Time Serie of user: {id_user}")

    plt.show()



if __name__ == '__main__':

    baseline = sys.stdin
    eda_baseline_boxplot_mean(baseline)