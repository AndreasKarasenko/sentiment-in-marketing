# Utils
We provide multiple util functions that can be reused elsewhere.
The following provides a brief description of some aspects.

## Table of Contents
1. [Eval](#eval)
2. [Describe](#describe)

## [eval](eval.py)
We provide a short script for eval metrics. Since we are in a text classification scenario, we are interested in [accuracy](#accuracy), [precision](#precision), [recall](#recall), and the [F1 score](#f1-score). It expects an array of true labels and an array of predicted labels. It has an optional str parameter (weighting) for the averaging method. More on averaging [here](../examples/metrics/README.md).


### Accuracy
Accuracy is a popular metric for classification tasks, but only if the class distribution is not skewed. In cases of extreme class imbalance accuracy should not be used.

$$Accuracy = \frac{TP + TN}{TP + FP + FN + TN}$$

### Precision
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. High precision relates to the low false positive rate.

$$Precision = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)
Recall is the ratio of correctly predicted positive observations to the all observations in actual class. High recall relates to the low false negative rate.

$$Recall = \frac{TP}{TP + FN}$$

### F1 Score
F1 Score is the weighted average of Precision and Recall. It tries to find the balance between precision and recall.

$$F1Score = \frac{2 * (Recall * Precision)}{Recall + Precision}$$

## [describe.py](describe.py)
We provide a helper function that offers statistical description of a dataset. descripe.py should cover most relevant statistics.

## [save_results.py](save_results.py)
This function saves results of a HPO experiment 