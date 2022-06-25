import numpy as np
from sklearn.metrics import confusion_matrix
import re
import torch.nn as nn

class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass

class ConfusionMatrix(Metric):

    def __init__(self, labels, ignore_class=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_class = ignore_class  # the class index to be removed
        self.labels = labels
        self.n_classes = len(self.labels)
        if self.ignore_class is not None:
            self.matrix = np.zeros((self.n_classes-1, self.n_classes-1))
        else:
            self.matrix = np.zeros((self.n_classes, self.n_classes))

        # self.pred = []
        # self.targ = []

    def get_labels(self):
        if self.ignore_class is not None:
            return np.delete(self.labels, self.ignore_class)
        return self.labels

    def forward(self, y_pr, y_gt):
        # sklearn.metrics
        pred = y_pr.view(-1).cpu().detach().tolist()
        targ = y_gt.view(-1).cpu().detach().tolist()

        # To format the matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        # confusion_matrix(y_true, y_pred)
        # array([[2, 0, 0],  # two zeros were predicted as zeros
        #        [0, 0, 1],  # one 1 was predicted as 2
        #        [1, 0, 2]])  # two 2s were predicted as 2, and one 2 was 0
        matrix = confusion_matrix(targ, pred, labels=self.labels)

        if self.ignore_class is not None:
            matrix = np.delete(matrix, self.ignore_class, 0)  # remove the row
            matrix = np.delete(matrix, self.ignore_class, 1)  # remove the column

        self.matrix = np.add(self.matrix, matrix)

        results_vec = {"labels": self.get_labels(), "confusion matrix": self.matrix}

        total = np.sum(self.matrix)
        true_positive = np.diag(self.matrix)
        sum_rows = np.sum(self.matrix, axis=0)
        sum_cols = np.sum(self.matrix, axis=1)
        false_positive = sum_rows - true_positive
        false_negative = sum_cols - true_positive
        # calculate accuracy
        overall_accuracy = np.sum(true_positive) / total
        results_scalar = {"OA": overall_accuracy}

        # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        p0 = np.sum(true_positive) / total
        pc = np.sum(sum_rows * sum_cols) / total ** 2
        kappa = (p0 - pc) / (1 - pc)
        results_scalar["Kappa"] = kappa

        # Per class recall, prec and F1
        recall = true_positive / (sum_cols + 1e-12)
        results_vec["R"] = recall
        precision = true_positive / (sum_rows + 1e-12)
        results_vec["P"] = precision
        f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)
        results_vec["F1"] = f1

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
            results_vec["IoU"] = iou
            results_scalar["mIoU"] = np.nanmean(iou)

        # Per class accuracy
        cl_acc = true_positive / (sum_cols + 1e-12)
        results_vec["Acc"] = cl_acc

        # weighted measures
        prob_c = sum_rows / total
        prob_r = sum_cols / total
        recall_weighted = np.inner(recall, prob_r)
        results_scalar["wR"] = recall_weighted
        precision_weighted = np.inner(precision, prob_r)
        results_scalar["wP"] = precision_weighted
        f1_weighted = 2 * (recall_weighted * precision_weighted) / (recall_weighted + precision_weighted)
        results_scalar["wF1"] = f1_weighted
        random_accuracy = np.inner(prob_c, prob_r)
        results_scalar["RAcc"] = random_accuracy

        return results_vec, results_scalar