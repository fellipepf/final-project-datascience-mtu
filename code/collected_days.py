import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import  utils

#Code to investigate the values from the file days.csv


def loadFileCSV(dir):
    data = pd.read_csv(dir, delimiter=',')

    return data


def create_structure():
    dir_collected_days = "../psykose/days.csv"

    content_collected_days = loadFileCSV(dir_collected_days)


    print(dir_collected_days)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

    return content_collected_days

# the values in the column "id" has string with a sequential number.
# this function split the string and extracts only the value that identify the type of user
# and then create a new column with the class to be easier count the values
def prepare_dataset(collected_days):
    for index, row in collected_days.iterrows():
        id = row['id']

        x = id.split("_")
        if x[0].lstrip() == "patient":
            user_type = utils.Target.PATIENT
        else:
            user_type = utils.Target.CONTROL
        collected_days.loc[index,'class'] = user_type

    return collected_days

def stats_collected_days(collected_days):

    desc = collected_days.groupby('class').describe()
    print(desc)

    #control = baseline.loc[baseline['class'] == 0]

    #sns.set_style("darkgrid")
    #sns.boxplot(x=baseline["class"], y=baseline["f.mean"])
    #plt.show()

    #plt.show()
    #print(collected_days)
    #print(control)


if __name__ == '__main__':
    collected_days = create_structure()
    collected_days = prepare_dataset(collected_days)
    stats_collected_days(collected_days)
    #print(collected_days.head())

