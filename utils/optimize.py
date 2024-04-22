### a script that takes json config files from ./config/ and uses gridsearchcv to find the best hyperparameters for each model and saves the optimal parameters to ./config/optimized
# Path: utils/optimize.py
# Import necessary libraries
import argparse
import json
import os
import time
import warnings

#
import pandas as pd

# from cuml.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import FunctionTransformer
from skopt import BayesSearchCV

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics


def label_transformer(y):
    """Transform the labels to start from 0."""
    y_transformer = FunctionTransformer(lambda y: y - 1, validate=False)
    y = y_transformer.transform(y)
    return y


def run_bayesian_optimization(
    model,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    verbose: int = 1,
    n_jobs: int = 1,
    n_iter: int = 32,
):
    """Run bayesian optimization on a model.

    Args:
        model: a sklearn model
        params: a dictionary of hyperparameters
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
    """

    y_train = label_transformer(y_train)
    y_test = label_transformer(y_test)

    pipeline = Pipeline([("tranform", TfidfVectorizer()), ("clf", model)])
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision_macro",
        "Recall": "recall_macro",
        "F1": "f1_macro",
    }
    opt = BayesSearchCV(
        pipeline,
        params,
        n_iter=n_iter,
        cv=5,
        verbose=verbose,
        scoring=scoring,
        refit="F1",
        return_train_score=True,
        n_jobs=n_jobs,
        n_points=4,
    )

    # Fit the model
    print("Fitting the model")
    opt.fit(X_train, y_train)

    # Make predictions
    predictions = opt.predict(X_test)

    # Print the evaluation metrics
    metrics = eval_metrics(y_test, predictions)

    return opt, metrics


def run_gridsearchcv(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    verbose=1,
    n_jobs: int = 1,
    mode: str = "cpu",
):
    """Run grid search cross validation on a model.

    Args:
        model: a sklearn model
        params: a dictionary of hyperparameters
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
    """
    y_train = label_transformer(y_train)
    y_test = label_transformer(y_test)
    pipeline = Pipeline([("transformer", TfidfVectorizer()), ("clf", model)])
    if mode == "gpu":
        pipeline = Pipeline(
            [
                ("vect", HashingVectorizer()),
                ("clf", model),
            ]
        )

    # define scores to be calculated
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision_macro",
        "Recall": "recall_macro",
        "F1": "f1_macro",
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=5,
        verbose=verbose,
        scoring=scoring,
        refit="F1",  # refit the model on the best F1 score
        return_train_score=True,
        n_jobs=n_jobs,
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Make predictions
    predictions = grid_search.predict(X_test)

    # Print the evaluation metrics
    metrics = eval_metrics(y_test, predictions)

    return grid_search, metrics
