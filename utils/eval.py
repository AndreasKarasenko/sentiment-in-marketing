### evaluation functions for multi-class classification

from typing import Union, List
import numpy as np
import sklearn.metrics as metrics


def eval_metrics(
    y_true: Union[np.ndarray, List[int], List[float]],
    y_pred: Union[np.ndarray, List[int], List[float]],
    weighting: str = "weighted",
):
    """Compute evaluation metrics for multi-class classification.

    Args:
        y_true: true labels
        y_pred: predicted labels
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=weighting)
    precision = metrics.precision_score(y_true, y_pred, average=weighting)
    recall = metrics.recall_score(y_true, y_pred, average=weighting)
    report = metrics.classification_report(y_true, y_pred)
    return acc, f1, precision, recall, report
