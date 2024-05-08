# main script to eval all normal machine learning based models
# NB, LR, SVM, DT, RF, XGBoost

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
from models import MODELS

### import evaluation functions
from utils.dataloader import get_config_names, load_samples
from utils.eval import eval_metrics
from utils.optimize import run_bayesian_optimization, run_gridsearchcv
from utils.preprocess import TextPreprocessor
from utils.save_results import save_results

# warnings.filterwarnings("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"

### and parse the cli arguments


def run_eval(
    model_dict: Dict[str, Callable[[], Any]],
    args: argparse.Namespace,
    tuning: str = "grid",
):
    print("running eval")
    datasets, input_vars, target_vars = get_config_names(args)  # get the config names
    for index, name in enumerate(datasets):

        X_train, X_test, y_train, y_test = load_samples(
            name, input_vars, target_vars[0], args
        )
        # get the model
        for model_name, model_func in model_dict.items():
            model_instance = model_func()
            if tuning == "grid":
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
            elif tuning == "bayes":
                start = time.time()
                grid, metrics = run_bayesian_optimization(
                    model_instance,
                    search_space[model_name]["hyperparameters"],
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    verbose=args.verbose,
                    n_jobs=args.njobs,
                )
            else:
                raise ValueError("tuning must be either grid or bayes")
            end = time.time()
            walltime = end - start
            print(grid.best_params_)
            print(metrics[-1])
            filename = (
                model_name
                + "_"
                + name
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            save_results(filename, model_name, name, metrics, args, walltime, grid)
    return 1


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
    run_eval(MODELS, args, tuning="grid")
