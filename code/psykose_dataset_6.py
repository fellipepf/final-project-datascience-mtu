
import numpy as np
import pandas as pd
import glob
from enum import Enum
import os
import log_configuration
from collected_days import DaysCollected

class Target(Enum):
    PATIENT = 1
    CONTROL = 0


class LoadDataset:

    def __init__(self):
        self.logger = log_configuration.logger

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
        self.logger.info('Loading files...')

        dir_control = "./psykose/control/*.csv"
        # dir_control = "/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/psykose/control/*.csv"
        dir_patient = "./psykose/patient/*.csv"
        # dir_patient = "/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/psykose/patient/*.csv"

        contentControl = self.loadFileCSV(dir_control, Target.CONTROL)
        contentPatient = self.loadFileCSV(dir_patient, Target.PATIENT)

    #print(contentControl)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

        return contentControl, contentPatient

    def get_dataset_by_class(self):
        return self.control, self.patient

    def get_dataset_joined(self):
        #return self.control | self.patient  #python 3.9
        return dict(self.control, **self.patient)

#include methods to limit values by number of days
# transform in only one dataset


class PreProcessing:

    def __init__(self, *args):
        self.logger = log_configuration.logger

        if len(args) == 1:
            self.control_patient = args[0]

        if len(args) == 2:
            self.control_patient = args[0] | args[1]

        #Dictinary structure
        #filter dataset by number of days
        self.control_patient_byday = self.__process_dataset_byday(self.control_patient)

        #split by time perio

        #Dataframe structure
        #transform dataset from dict to just one dataframe
        self.df_dataset =  self.create_one_df_struture(self.control_patient_byday)

        self.df_dataset_category = self.df_structure_category(self.df_dataset)


    def __process_dataset_byday(self, dataset):
        '''
        This function reduce the dataset to the number of days for each person
        '''

        self.logger.info('Split dataset by day...')

        dic_result = dict()
        remove_fist_day = True

        days_collected = DaysCollected()
        days = days_collected.get_days_collected()

        for key in set(dataset.keys()).difference({'target'}):
            value = dataset[key]

            df = value['timeserie']
            user_class = value['target']

            df["datetime"] = pd.to_datetime(df["timestamp"])
            # group_day = df.groupby(df["datetime"].dt.day)['activity']

            n_days_limit = days.loc[days['id'] == key]['days'].item()
            group_day = df.groupby(pd.Grouper(key='datetime', freq='D'))

            if remove_fist_day:
                list_days = list(group_day)

                # remove first day and
                # slice n number of elements defined
                group_n_days = list_days[1:n_days_limit + 1]
            else:
                group_n_days = list(group_day)[:n_days_limit]

            # get the second element from tuple in a list using comprehension list
            df_days = [tuple[1] for tuple in group_n_days]

            # df_result.extend(group_n_days)

            # transform list of dataframes to only one dataframe
            # df_all = pd.concat(df_days)
            # df_all['class'] = user_class

            dic_result[key] = {}
            dic_result[key]['timeserie'] = df_days
            dic_result[key]['user_class'] = user_class

        return dic_result


    def create_one_df_struture(self, control_patient_byday):
        self.logger.info('Creating dataframe with all values...')

        df_data_set = pd.DataFrame()

        for key, value in control_patient_byday.items():
            df_ts = value['timeserie']
            df_ts = pd.concat(df_ts)

            df_ts['user'] = key
            df_ts['class'] = value['user_class']

            df_data_set = df_data_set.append(df_ts, ignore_index=True )

        return df_data_set

    def df_structure_category(self, df_data_set):
        self.logger.info('Creating dataframe with period of time categories...')

        df_data_set = df_data_set.set_index('datetime')
        df_data_set['category'] = None

        #morning
        #morning = df_data_set.between_time('0:00', '6:59')
        #df_data_set.loc[morning, 'category'] = 0
        #df_data_set['category'][(df_data_set.index.hour >= 0) & (df_data_set.index.hour < 7)] = 0
        df_data_set.loc[(df_data_set.index.hour >= 0) & (df_data_set.index.hour < 4),'category'] = 0

        df_data_set.loc[(df_data_set.index.hour >= 4) & (df_data_set.index.hour < 8),'category'] = 1

        df_data_set.loc[(df_data_set.index.hour >= 8) & (df_data_set.index.hour < 12),'category'] = 2

        df_data_set.loc[(df_data_set.index.hour >= 12) & (df_data_set.index.hour < 16),'category'] = 3

        #afternoon
        df_data_set.loc[(df_data_set.index.hour >= 16) & (df_data_set.index.hour < 20), 'category'] = 4

        #night
        df_data_set.loc[(df_data_set.index.hour >= 20) & (df_data_set.index.hour < 24), 'category'] = 5

        #n = df_data_set[df_data_set['category'] == None ]

        return df_data_set


