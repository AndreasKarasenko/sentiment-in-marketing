### A function to load the data from data/samples/train.csv and data/samples/test.csv
from typing import List
import pandas as pd
import os


def load_data(config: dict, path: List[str]):
    """Load the data from data/samples/train.csv and data/samples/test.csv
    args:
        config: config object with information about input / output variables
        path: path to the data
    return: train and test dataframes"""

    # Load train and test data
    train = pd.read_csv(path)
    test = pd.read_csv(path)

    train = train[config["input_variables"] + config["output_variables"]]
    test = test[config["input_variables"] + config["output_variables"]]

    return train, test
