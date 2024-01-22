## Evaluation options
### Simple comparison

We can simply compare the performance of our classifiers and choose the best.
This is however oftentimes criticized as it does not reflect the stability of the model. Retraining may lead to wildly different results. Instead other approaches should be considered.

### Statistical tests

Statistical tests like the T-Test or Friedman Test are often used to compare the performance of classifiers. Using these tests, we can determine if the differences in performance between classifiers are statistically significant. 

A low p-value (typically below 0.05) indicates that the differences in performance are statistically significant. However, these tests should be used as a guide and not as definitive proof, as they can be influenced by factors such as the choice of dataset and the number of samples.