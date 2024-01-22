from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils.eval_statistic import run_friedman_test, run_posthoc
import numpy as np

# Load the iris dataset
iris = load_iris()

# Define the classifiers
classifiers = [DecisionTreeClassifier(), SVC(probability=True), RandomForestClassifier()]

# Define the metrics
metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_ovr']

# Store the results of the classifiers
results = []

# For each classifier
for clf in classifiers:
    # For each metric
    for metric in metrics:
        # Perform cross-validation and store the results
        scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring=metric)
        results.append(scores)

# Print the results
print(results)

# Run the Friedman test on the results
statistic, p_value = run_friedman_test(*results)


print(f"Test statistic: {statistic}")
print(f"P-value: {p_value}")


# Run the Friedman test on the results
statistic, p_value = run_friedman_test(*results)

print(f"Friedman Test statistic: {statistic}")
print(f"Friedman Test P-value: {p_value}")

# Run the post-hoc Nemenyi test on the results
posthoc = run_posthoc(np.array(results).T)

print(posthoc) # TODO fix this, currently bad results probably because of wrong results format