### evaluation functions for multi-class classification


import numpy as np
import sklearn.metrics as metrics


def eval_metrics(y_true, y_pred):
    """Compute evaluation metrics for multi-class classification.

    Args:
        y_true: true labels
        y_pred: predicted labels
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")
    precision = metrics.precision_score(y_true, y_pred, average="weighted")
    recall = metrics.recall_score(y_true, y_pred, average="weighted")
    report = metrics.classification_report(y_true, y_pred)
    return acc, f1, precision, recall, report

def weighted_score(probs):
    """calculates the weighted score as a sum of
    class probabilites multiplied by their respective index + 1
    Example: porbs[0] = [0.1, 0.2, 0.7] -> weighted_score = 0.1*1 + 0.2*2 + 0.7*3"""
    
    