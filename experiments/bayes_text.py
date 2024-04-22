# short experiment to run bayesian optimization with text data
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer as skHV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skopt import BayesSearchCV

twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    twenty_train.data, twenty_train.target, test_size=0.25, random_state=0
)
print(X_train[0])
cv = skHV()

X_train = skHV().fit_transform(X_train).astype(np.float32)
X_test = skHV().transform(X_test).astype(np.float32)

y_train = np.array(y_train).astype(np.int32)
y_test = np.array(y_test).astype((np.int32))

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision_macro",
    "Recall": "recall_macro",
    "F1": "f1_macro",
}
pipeline = Pipeline([("tranform", skHV()), ("clf", SVC())])
opt = BayesSearchCV(
    SVC(),
    {
        "C": (1e-6, 1e6, "log-uniform"),
        "gamma": (1e-6, 1e1, "log-uniform"),
        "degree": (1, 8),  # integer valued parameter
        "kernel": ["linear", "poly", "rbf"],  # categorical parameter
    },
    n_iter=1,
    cv=3,
    verbose=3,
    n_jobs=-1,
    n_points=4,
    scoring=scoring,
    refit="F1",
)


start = time.time()
opt.fit(X_train, y_train)
end = time.time()

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print("took %s seconds" % (end - start))
print("best params: %s" % opt.best_params_)
