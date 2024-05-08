# main script to eval all normal machine learning based models
# NB, LR, SVM, DT, RF, XGBoost

import argparse
import json
import os
import time

# import utilities
import warnings
from datetime import datetime

import pandas as pd

### import the configurations
from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.eval import eval_metrics
from utils.optimize import run_gridsearchcv
from utils.save_results import save_results

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

import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

### import preprocessing functions
from sklearn.base import TransformerMixin

### import the models dict
from models import MODELS


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


if __name__ == "__main__":
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv")
        train.dropna(inplace=True)
        test = pd.read_csv(args.data_dir + i + "_test.csv")
        test.dropna(inplace=True)

        X_train = train[input_vars]
        X_test = test[input_vars]
        # apply the text preprocessor to the text data
        # If we don't do this we incur a 4s penalty for each fold
        X_train = TextPreprocessor().transform(X_train)
        X_test = TextPreprocessor().transform(X_test)

        y_train = train[target_vars]
        y_test = test[target_vars]

        # get the model
        for model_name, model_func in MODELS.items():
            model_instance = model_func()
            start = time.time()
            grid, metrics = run_gridsearchcv(
                model_instance,
                search_space[model_name]["hyperparameters"],
                X_train,
                y_train,
                X_test,
                y_test,
                verbose=args.verbose,
                n_jobs=args.njobs,
            )
            end = time.time()
            walltime = end - start
            print(grid.best_params_)
            print(metrics[-1])
            filename = (
                model_name
                + "_"
                + i
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            save_results(filename, model_name, i, metrics, args, walltime, grid)
