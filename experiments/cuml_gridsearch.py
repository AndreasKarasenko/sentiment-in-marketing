# This script is an example of gridsearch using sklearn pipeline and gridsearchcv with cuml
import os
from re import X
import time
import joblib
import cudf
import cuml
import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from cuml.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from cuml.ensemble import RandomForestClassifier


import warnings
import argparse
import json
from datetime import datetime

from utils.optimize import run_gridsearchcv
from utils.save_results import save_results

### import evaluation functions
from utils.eval import eval_metrics

### import the configurations
from config.utils_config.argparse_args import arguments

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

### and parse the cli arguments
parser = argparse.ArgumentParser(
    description="Finds the best hyperparameters for the models"
)
for arg in arguments:
    parser.add_argument(
        arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
    )

args = parser.parse_args()

### load the search space json
search_space = json.load(open(args.config_dir, "r"))

### get the dataset names
datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
datasets = datasets["datasets"]
print(datasets)

### get the input and target vars
input_vars = json.load(open(args.data_config + "input_config.json", "r"))
input_vars = input_vars["input_var"]

target_vars = json.load(open(args.data_config + "target_config.json", "r"))
target_vars = target_vars["target_vars"]

### import preprocessing functions
from sklearn.base import TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string


# create a text preprocessor class to preprocess the text data once before the hp tuning
# since the preprocessor does not learn there is no risk of overfitting
# all other transformation is still done in folds
class TextPreprocessor(TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = []
        for document in X:
            # Lowercase
            document = document.lower()
            # Remove punctuation
            document = document.translate(str.maketrans("", "", string.punctuation))
            # Tokenize
            words = word_tokenize(document)
            # Lemmatize
            words = [self.lemmatizer.lemmatize(word) for word in words]
            X_transformed.append(" ".join(words))
        return X_transformed


for i in datasets:
    train = pd.read_csv(args.data_dir + i + "_train.csv")
    train.dropna(inplace=True)
    # train = cudf.from_pandas(train)
    test = pd.read_csv(args.data_dir + i + "_test.csv")
    test.dropna(inplace=True)
    # test = cudf.from_pandas(test)
    
    X_train = train[input_vars]
    X_test = test[input_vars]
    print(X_train)
    print(type(X_train))
    # apply the text preprocessor to the text data
    # If we don't do this we incur a 4s penalty for each fold
    # X_train = TextPreprocessor().transform(X_train)
    # X_test = TextPreprocessor().transform(X_test)
    
    y_train = train[target_vars]
    y_test = test[target_vars]
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestClassifier()),
        ]
    )
    from sklearn.metrics import classification_report
    
    params = {
        "clf__n_estimators": [10, 100],
        "clf__max_depth": [10, 100],
    }
    
    grid = GridSearchCV(pipeline, params, cv=2)
    grid.fit(X_train, y_train)


# import cuml
# from sklearn import datasets
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split, GridSearchCV
# from cuml.neighbors import KNeighborsClassifier
# X, y = datasets.make_classification(
#     n_samples=100
# )
# print(type(X))
# pipe = Pipeline([
#         ('normalization', MinMaxScaler()),
#         ('classifier', KNeighborsClassifier(metric='euclidean', output='input'))
# ])

# parameters = {
#     'classifier__n_neighbors': [1,3,6] 
# }
# grid_search = GridSearchCV(pipe, parameters, cv=2)
# grid_search.fit(X, y)
# GridSearchCV(cv=2,
#              estimator=Pipeline(steps=[('normalization', MinMaxScaler()),
#                                        ('classifier', KNeighborsClassifier())]),
#              param_grid={'classifier__n_neighbors': [1, 3, 6]})