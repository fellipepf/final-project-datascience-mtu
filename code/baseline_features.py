import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

'''
Code to investigate the values from the file schizophrenia-features.csv

'''
class BaselineFeatures:
    def __init__(self):
        self.baseline = self.create_structure()

    def loadFileCSV(self, dir):
        data = pd.read_csv(dir, delimiter=',')

        return data


    def create_structure(self):
        #full file path to be able show the graphs on notebook in another folder
        dir_baseline = "/Users/fellipeferreira/OneDrive/CIT - Master Data Science/Semester 3/project/final-project-datascience-mtu/psykose/schizophrenia-features.csv"

        content_baseline = self.loadFileCSV(dir_baseline)

        #structureControl = pd.DataFrame(zip(contentControl), columns=["data"])

        return content_baseline

    '''
     FUNCTION stats_baseline_boxplot

    '''
    def stats_baseline_boxplot(self):
        baseline = self.baseline
        control = baseline.loc[baseline['class'] == 0]

        sns.set_style("darkgrid")
        sns.boxplot(x=baseline["class"], y=baseline["f.mean"])
        plt.show()

        #plt.show()


    def control_statistics(self):
        baseline = self.baseline
        control = baseline.loc[baseline['class'] == 0]
        print(control.describe())

    def patient_statistics(self):
        baseline = self.baseline
        patient = baseline.loc[baseline['class'] == 1]
        print(patient.describe())

    def get_baseline(self):
        return self.baseline



if __name__ == '__main__':
   # baseline = create_structure()
   # stats_baseline_boxplot(baseline)

   bf = BaselineFeatures()
   bf.stats_baseline_boxplot()
   bf.control_statistics()

