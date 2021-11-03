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

#preprocessing and data cleaning
def preprocessing(patient_info):
    for index, row in patient_info.iterrows():
        age_range = row['age']
        age = age_one_value(age_range)

        patient_info.loc[index, 'new_age'] = age
    return patient_info

def age_one_value(age_range):
    age_splited = age_range.split("-")

    age_left = age_splited[0]
    age_right = age_splited[1]

    avg = (int(age_left) + int(age_right))/2

    return avg


def mean_age_by_gender(patient_info):
    graph = patient_info.groupby('gender').new_age.mean().plot.bar()
    graph.set_title("Gender age")
    graph.set_xlabel("gender")
    graph.set_ylabel("Mean age")
    graph.bar_label(graph.containers[0])
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.show()


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
    patient_info = preprocessing(patient_info)

    mean_age_by_gender(patient_info)
    stats_male_female(patient_info)