# Utils
We provide multiple util functions that can be reused elsewhere.
The following provides a brief description of some aspects.

## Table of Contents
1. [Eval](#eval)
2. [Describe](#describe)
3. [Save results](#save_results)
4. [Summarize Results](#summarize_results)

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

## [describe](describe.py)
We provide a helper function that offers statistical description of a dataset. descripe.py should cover most relevant statistics.

## [save_results](save_results.py)
This function automatically saves results of a HPO experiment as well as the corresponding model as a pkl file.
It creates a nested json file containing the model name, the dataset, the overall time of the experiment, metrics,
best hyperparameters, the cli and config arguments used, the search space, the best score, and more. A folder is created using the model name (if it does not exist already) and the json / pkl files are saved in that location. Each file is named using the model, dataset, and date for easy filtering.

It is not intended to use this function outside of the main files.

## [summarize_results](summarize_results.py)
This utility script summarizes the results of your experiments. After you ran your HPO experiments using one of the main scripts, [save_results](#save_results) will have created multiple json and pkl files. This script retrieves the json files for each model and considers only the NEWEST experiment per dataset. You can then pass an optional parameter called ```mode``` which accepts "all" or "average". If you chose "all" you might also want to pass the ```metric``` flag to change which metric gets saved (ACC, F1, PREC, REC).

If you choose "average" you will get the Acc, F1, Prec, Rec values for each model, averaged over all datasets. If you choose "all" you get all F1 scores for a model for each dataset. The corresponding tables are then save to [results/overview](../results/overview/) for easy copying.

#### Example usage
Remember to run these commands from the [top level location](../) in your project.
~~~bash
python -m utils.summarize_results --mode "average"
python -m utils.summarize_results --mode "all"
python -m utils.summarize_results --mode "all" --metric "F1"
~~~