# short experiment to compare bayesian optimization with grid search
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt import BayesSearchCV

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=0.25, random_state=0
)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SVC(),
    {
        "C": (1e-6, 1e6, "log-uniform"),
        "gamma": (1e-6, 1e1, "log-uniform"),
        "degree": (1, 8),  # integer valued parameter
        "kernel": ["linear", "poly", "rbf"],  # categorical parameter
    },
    n_iter=32,
    cv=3,
)
start = time.time()
opt.fit(X_train, y_train)
end = time.time()

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print("took %s seconds" % (end - start))
print("best params: %s" % opt.best_params_)
