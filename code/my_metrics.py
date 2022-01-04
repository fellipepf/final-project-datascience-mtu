import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    classification_report, matthews_corrcoef


class ModelMetrics:
    def __init__(self, y_true, y_preds):
        self.y_true = y_true
        self.y_preds = y_preds

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_preds)

    def precision(self):
        return precision_score(self.y_true, self.y_preds, average=None)

    def average_precision_score(self):
        return metrics.average_precision_score(self.y_true, self.y_preds)

    def recall(self):
        return recall_score(self.y_true, self.y_preds)

    def f1_score(self):
        return f1_score(self.y_true, self.y_preds)

    def precision_recall_fscore_support(self):
        return precision_recall_fscore_support(self.y_true, self.y_preds)

    def classification_report(self):
        return classification_report(self.y_true, self.y_preds)

    def auc(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_preds)
        return metrics.auc(fpr, tpr)

    def matthews_corrcoef(self):
        return matthews_corrcoef(self.y_true, self.y_preds)

    def all_metrics(self):
        all_metrics = {}
        all_metrics["accuracy"] = self.accuracy()
        all_metrics["precision"] = self.precision()
        all_metrics["average_precision"] = self.average_precision_score()
        all_metrics["recall"] = self.recall()
        all_metrics["f1_score"] = self.f1_score()
        all_metrics["precision_recall_fscore_support"] = self.precision_recall_fscore_support()
        all_metrics["classification_report"] = self.classification_report()
        all_metrics["auc"] = self.auc()
        all_metrics["matthews_corrcoef"] = self.matthews_corrcoef()

        return all_metrics

if __name__ == '__main__':
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    mm = ModelMetrics(y_true, y_scores)
    all = mm.all_metrics()
    print(all)