import numpy as np
import sklearn.metrics as metrics

# example 1: Binary imbalance
y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# f1 score
f1 = metrics.f1_score(y_true, y_pred, average="binary")
print(f"Binary F1 score: {f1:.2f}")

# example 2: imbalance and weighted f1 score
f1 = metrics.f1_score(y_true, y_pred, average="weighted")
print(f"Imbalanced Weighted F1 score: {f1:.2f}")

# example 3: multi-class classification
y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# f1 score
f1 = metrics.f1_score(y_true, y_pred, average="macro")
print(f"Macro multi-class F1 score: {f1:.2f}")

f1 = metrics.f1_score(y_true, y_pred, average="micro")
print(f"Micro multi-class F1 score: {f1:.2f}")

f1 = metrics.f1_score(y_true, y_pred, average="weighted")
print(f"Weighted multi-class F1 score: {f1:.2f}")

# as the number of classes increases, the weighted f1 score becomes more sensitive to class imbalance
# this is because the weighted f1 score is the average of the f1 scores for each class, weighted by the number of true instances in each class
# in the case of class imbalance, the weighted f1 score will be biased towards the majority class
# see e.g.
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
f1 = metrics.f1_score(y_true, y_pred, average="weighted")
print(f"Weighted multi-class F1 score (heavily biased): {f1:.2f}")

# we can also demonstrate the effect of class imbalance on the weighted f1 score by using the eval_metrics function
# we can see that the weighted f1 score is biased towards the majority class
# this is also reflected in the precision and recall scores
def eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, weighting: str = "binary"
):
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

for i in eval_metrics(y_true, y_pred):
    print(i)