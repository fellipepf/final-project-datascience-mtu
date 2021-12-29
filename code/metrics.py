from sklearn import metrics
from sklearn.metrics import accuracy_score


class ModelMetrics:
    def __init__(self, y_true, y_preds):
        self.y_true = y_true
        self.y_preds = y_preds

    def accuracy(self):
        self.accuracy_metric = accuracy_score(self.y_test, self.y_preds)
