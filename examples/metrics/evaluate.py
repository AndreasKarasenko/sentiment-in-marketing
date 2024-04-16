import numpy as np
import sklearn.metrics as metrics


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute evaluation metrics for multi-class classification.

    Args:
        y_true: true labels
        y_pred: predicted labels
        weighting: type of averaging for multi-class classification
    """
    f1 = []
    precision = []
    recall = []
    for i in ("macro", "micro", "weighted"):
        f1.append(metrics.f1_score(y_true, y_pred, average=i))
        precision.append(metrics.precision_score(y_true, y_pred, average=i))
        recall.append(metrics.recall_score(y_true, y_pred, average=i))
    acc = metrics.accuracy_score(y_true, y_pred)
    return acc, f1, precision, recall
