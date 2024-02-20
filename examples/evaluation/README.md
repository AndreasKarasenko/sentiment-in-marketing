## Evaluation options
### Simple comparison

We can simply compare the performance of our classifiers and choose the best.
This is however oftentimes criticized as it does not reflect the stability of the model. Retraining may lead to wildly different results. Instead other approaches should be considered.

### Statistical tests

Statistical tests like the T-Test or Friedman Test are often used to compare the performance of classifiers. Using these tests, we can determine if the differences in performance between classifiers are statistically significant. 

A low p-value (typically below 0.05) indicates that the differences in performance are statistically significant. However, these tests should be used as a guide and not as definitive proof, as they can be influenced by factors such as the choice of dataset and the number of samples.

### Bootstrapping

Bootstrapping is a powerful statistical method for estimating the sampling distribution of an estimator by resampling with replacement from the original sample. It's often used to estimate the uncertainty of a model's performance.

In the context of classifier performance, bootstrapping can be used to create many subsets of the original data, train the classifier on these subsets, and then evaluate the performance on the remaining data. This process is repeated many times to get a distribution of performance metrics.

This method can provide a more robust measure of the model's performance, as it considers the variability of the model's performance over different subsets of data. However, like other statistical methods, bootstrapping has its limitations and assumptions, and should be used judiciously and in conjunction with other methods to make informed decisions about model performance.