import pandas as pd
import glob
import matplotlib.pyplot as plt

#Code to investigate values from the patients_info.csv
#This file contains only extra details for patients

def loadFileCSV(dir):
    data = pd.read_csv(dir, delimiter=',')

    return data


def create_structure():
    dir_patient_info = "../psykose/patients_info.csv"

    content_patient_info = loadFileCSV(dir_patient_info)


    print(content_patient_info)
    #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

    return content_patient_info

def stats_male_female(patient_info):
    #patient_info.groupby(['gender']).count().plot(kind='bar')
    graph = patient_info['gender'].value_counts().plot.bar()
    graph.set_title("Gender distribution")
    graph.set_xlabel("gender")
    graph.set_ylabel("quantity")
    graph.bar_label(graph.containers[0])
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.show()



if __name__ == '__main__':
    patient_info = create_structure()
    stats_male_female(patient_info)