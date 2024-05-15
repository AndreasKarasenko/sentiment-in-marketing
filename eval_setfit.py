# example script to run and evaluate setfit based few shot learning on sentiment data
# unlike zero or few shot prompting setfit first learns a sentence transformer base.
# This base essentially learns to  differentiate sentences that belong to the same class
# from those that belong to different classes.

import argparse
import json
import os
import time

# import utilities
import warnings
from datetime import datetime

# typing
from typing import Any, Callable, Dict, List

import pandas as pd

### import the configurations
from config.utils_config.argparse_args import arguments

### import the models dict
from setfit import sample_dataset, SetFitModel, SetFitTrainer
from datasets import Dataset

### import evaluation functions
from utils.dataloader import get_config_names, load_samples
from utils.eval import eval_metrics
from utils.optimize import run_gridsearchcv
from utils.preprocess import TextPreprocessor
from utils.save_results import save_results

# warnings.filterwarnings("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"

### and parse the cli arguments

# TODO use this paper to motivate https://arxiv.org/pdf/2308.14634


if __name__ == "__main__":
    print("test")
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

    print("input_vars", input_vars)
    print("target_vars", target_vars)
    
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv")
        test = pd.read_csv(args.data_dir + "/subsamples/" + i + "_test.csv")
        
        train = train.loc[:, [input_vars, target_vars[0]]]
        test = test.loc[:, [input_vars, target_vars[0]]]
        
        train.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        test.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        
        # assert that the labels start at 0
        if train.label.min() == 1:
            train.label -= 1
            test.label -= 1
        
        train_data = sample_dataset(Dataset.from_pandas(train), num_samples=8)
        print(train_data)
        raise ValueError("stop")