## Balancing Metrics
### Background

When calculating metrics for classification accuracy, the easiest scenario is a binary task. Is something class A or B? In this case the true and positive labels are easy to assign and calculate the metrics, but if there are more than 2 classes these break down. The [eval function](../../utils/README.md) we define goes into this a bit.

### Averaging approaches


Scikit-Learn offers a few approaches to average the metrics (from their documentation and specifically for precision):

1. Micro
    - Calculate metrics globally by counting the total true positives, false negatives and false positives.
2. macro
    - Calculate metrics for each label, and find their unweighted mean.  This does not take label imbalance into account.
3. weighted
    - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

Depending on the distribution of the class labels some averaging methods can be overly optimistic. If strong imbalance is present and weighted averaging is used, a classifier with poor performance on the minority class and good performance on the majority class will report a good metric.

There's also the option of "samples" which we will not go into here.