import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    classification_report, matthews_corrcoef


class ModelMetrics:
    def __init__(self, y_true, y_preds):
        self.y_true = y_true
        self.y_preds = y_preds

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_preds.round())

    def precision(self):
        return precision_score(self.y_true, self.y_preds.round(), average="binary")

    def average_precision_score(self):
        '''
        This metric is used in the paper for Experiment I
        :return:
        '''
        return metrics.average_precision_score(self.y_true, self.y_preds)

    def recall(self, average= "binary"):
        return recall_score(self.y_true, self.y_preds.round(), average = average)

    def f1_score(self, average= "binary"):
        return f1_score(self.y_true, self.y_preds.round(), average = average)

    def precision_recall_fscore_support(self):
        return precision_recall_fscore_support(self.y_true, self.y_preds.round())

    def classification_report(self):
        return classification_report(self.y_true, self.y_preds.round(), output_dict=True)

    def auc_roc(self):
        '''
        This metric is used in the paper for Experiment I
        :return:
        '''
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true, self.y_preds)
        return metrics.auc(fpr, tpr)

    def matthews_corrcoef(self, round=None):
        return matthews_corrcoef(self.y_true, self.y_preds.round())

    def confusion_matrix_rates(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_preds.round(), normalize='all').ravel()
        return tn, fp, fn, tp

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_preds.round())


    #may can cause error
    def all_metrics(self):
        all_metrics = {}
        all_metrics["accuracy"] = self.accuracy()
        all_metrics["precision"] = self.precision()
        all_metrics["average_precision"] = self.average_precision_score()
        all_metrics["recall"] = self.recall()
        all_metrics["f1_score"] = self.f1_score()
        all_metrics["precision_recall_fscore_support"] = self.precision_recall_fscore_support()
        all_metrics["classification_report"] = self.classification_report()
        all_metrics["auc_roc"] = self.auc_roc()
        all_metrics["matthews_corrcoef"] = self.matthews_corrcoef()

        return all_metrics

if __name__ == '__main__':
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    mm = ModelMetrics(y_true, y_scores)
    all = mm.all_metrics()
    print(all)