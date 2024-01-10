### a script that takes json config files from ./config/ and uses gridsearchcv to find the best hyperparameters for each model and saves the optimal parameters to ./config/optimized
# Path: utils/optimize.py
# Import necessary libraries
import warnings
import argparse
import os
import json
import pandas as pd
import time
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics

def run_gridsearchcv(model, params, X_train, y_train, X_test, y_test, verbose=1):
    """Run grid search cross validation on a model.

    Args:
        model: a sklearn model
        params: a dictionary of hyperparameters
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
    """
    y_transformer = FunctionTransformer(lambda y: y - 1, validate=False)
    y_train = y_transformer.transform(y_train)
    y_test = y_transformer.transform(y_test)
    pipeline = Pipeline(
        [("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", model)]
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
        cv=10,
        verbose=verbose,
        scoring=scoring,
        refit="F1",  # refit the model on the best F1 score
        return_train_score=True,
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Make predictions
    predictions = grid_search.predict(X_test)

    # Print the evaluation metrics
    metrics = eval_metrics(y_test, predictions)

    return grid_search, metrics