import numpy as np

thresh = 0
def f1_score(y_true, y_pred):
    """Calculate F1 score.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: F1 score.
    """
    tp = np.sum((y_true > 0) & (y_pred > thresh))
    fp = np.sum((y_true == 0) & (y_pred > thresh))
    fn = np.sum((y_true > 0) & (y_pred <= thresh))
    tn = np.sum((y_true == 0) & (y_pred <= thresh))
    return tp, fp, tn, fn

class Metrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.count = 0
        self.history = []

    def update(self, tp, fp, fn, tn):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.count += 1
        # f1 = None
        # try:
        #     f1 = self.f1_score()
        # except:
        #     pass
        # self.history.append((self.tp, self.fp, self.fn, self.tn, f1))

    def update_np(self, y_pred, y_true):
        if y_true.sum() == 0:
            return
        tp, fp, tn, fn = f1_score(y_pred, y_true)
        self.update(tp, fp, fn, tn)
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.count = 0
    
    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    def f1_score(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
    
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def region_similarity(self):
        p = self.precision()
        r = self.recall()
        return p / (1*p/r-p + 1)

    def get_all(self):
        return self.tp, self.fp, self.fn, self.tn
    
    def __str__(self) -> str:
        return "Precision: {:.4f}\nRecall: {:.4f}\nF1: {:.4f}\nAccuracy: {:.4f}\nRegion similarity: {:.4f}".format(
            self.precision(), self.recall(), self.f1_score(), self.accuracy(), self.region_similarity()
        )
    
    def print(self, precision=4):
        print("Precision: {}".format(round(self.precision(), precision)))
        print("Recall: {}".format(round(self.recall(), precision)))
        print("F1: {}".format(round(self.f1_score(), precision)))
        print("Accuracy: {}".format(round(self.accuracy(), precision)))
        print("Region similarity: {}".format(round(self.region_similarity(), precision)))
        print("")

    def table(self, precision=4):
        ret = [self.precision(), self.recall(), self.f1(), self.region_similarity()]
        return [round(r, precision) for r in ret]
    
