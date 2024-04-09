# script for hpo taken from https://docs.rapids.ai/deployment/stable/examples/xgboost-randomforest-gpu-hpo-dask/notebook/
import warnings

import dask_ml.feature_extraction

warnings.filterwarnings("ignore")  # Reduce number of messages/warnings displayed
import gzip
import os
import time
from urllib.request import urlretrieve

import cudf
import cuml
import cupy as cp
import dask
import dask.dataframe as dd
import dask_ml
import dask_ml.model_selection as dcv
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.dask.common import to_sparse_dask_array
from cuml.ensemble import RandomForestClassifier
from cuml.metrics.accuracy import accuracy_score
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from sklearn import datasets, tree
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer

# from dask_ml.model_selection import train_test_split
# from cuml.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# from cuml.ensemble


cluster = LocalCUDACluster()
client = Client(cluster)

# Load corpus
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
y = dask.array.from_array(twenty_train.target, asarray=False, fancy=False).astype(
    cp.int32
)
df = dd.from_pandas(
    pd.DataFrame(
        data={"text": twenty_train.data, "label": y},
        columns=["text", "label"],
    ),
    npartitions=25,
)
cv = dask_ml.feature_extraction.text.HashingVectorizer()
# X_train, X_test, y_train, y_test = train_test_split(df, "label", test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(
    df.text, df.label, test_size=0.2, shuffle=False
)
X_cpu = X_train
# X_cpu = X_train.to_pandas()
# X_cpu = cp.asarray(X_cpu)

y_cpu = y_train
# y_cpu = y_train.to_numpy()
# y_cpu = cp.asarray(y_cpu)

X_test_cpu = X_test
# X_test_cpu = X_test.to_pandas()
# X_test_cpu = cp.asarray(X_test_cpu)
y_test_cpu = y_test
# y_test_cpu = y_test.to_numpy()
# y_test_cpu = cp.asarray(y_test_cpu)

X_train = cv.fit_transform(X_train)
# X_train = cp.asarray(X_train)
# X_train = to_sparse_dask_array(X_train, client)
X_test = cv.transform(X_test)
# X_test = cp.asarray(X_test)

print(type(X_train))
print(type(X_test))

print(y_test_cpu)


# X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.2)
def accuracy_score_wrapper(y, y_hat):
    """
    A wrapper function to convert labels to float32,
    and pass it to accuracy_score.

    Params:
    - y: The y labels that need to be converted
    - y_hat: The predictions made by the model
    """
    y = y.astype("float32")  # cuML RandomForest needs the y labels to be float32
    return accuracy_score(y, y_hat, convert_dtype=True)


accuracy_wrapper_scorer = make_scorer(accuracy_score_wrapper)
cuml_accuracy_scorer = make_scorer(accuracy_score, convert_dtype=True)


def do_HPO(
    model, gridsearch_params, scorer, X, y, mode="gpu-Grid", n_iter=10, N_FOLDS=5
):
    """
    Perform HPO based on the mode specified
    mode: default gpu-Grid. The possible options are:
    1. gpu-grid: Perform GPU based GridSearchCV
    2. gpu-random: Perform GPU based RandomizedSearchCV
    n_iter: specified with Random option for number of parameter settings sampled
    Returns the best estimator and the results of the search
    """
    if mode == "gpu-grid":
        print("gpu-grid selected")
        clf = dcv.GridSearchCV(model, gridsearch_params, cv=N_FOLDS, scoring=scorer)
    elif mode == "gpu-random":
        print("gpu-random selected")
        clf = dcv.RandomizedSearchCV(
            model, gridsearch_params, cv=N_FOLDS, scoring=scorer, n_iter=n_iter
        )
    else:
        print("Unknown Option, please choose one of [gpu-grid, gpu-random]")
        return None, None
    res = clf.fit(X, y)
    print(
        "Best clf and score {} {}\n---\n".format(res.best_estimator_, res.best_score_)
    )
    return res.best_estimator_, res


def print_acc(model, X_train, y_train, X_test, y_test, mode_str="Default"):
    """
    Trains a model on the train data provided, and prints the accuracy of the trained model.
    mode_str: User specifies what model it is to print the value
    """
    y_pred = model.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_pred, y_test.astype("float32"), convert_dtype=True)
    print("{} model accuracy: {}".format(mode_str, score))


# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
model_gpu_xgb_ = xgb.XGBClassifier(tree_method="gpu_hist")
start = time.time()
print_acc(model_gpu_xgb_, X_train, y_cpu, X_test, y_test_cpu)
end = time.time()
print("Time taken for XGBoost on GPU: ", end - start)

model_gpu_xgb_ = xgb.XGBClassifier(tree_method="hist")
start = time.time()
print_acc(model_gpu_xgb_, X_train, y_cpu, X_test, y_test_cpu)
end = time.time()
print("Time taken for XGBoost on CPU: ", end - start)
