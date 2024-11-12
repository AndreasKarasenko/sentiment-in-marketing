# main script to run a naive baseline for all datasets

import argparse
import json
import time

# import utilities
from datetime import datetime
import numpy as np

# typing
from typing import Any, Callable, Dict


### import the configurations
from config.utils_config.argparse_args import arguments

### import the models dict

### import evaluation functions
from utils.dataloader import get_config_names, load_samples
from utils.optimize import run_bayesian_optimization, run_gridsearchcv
from utils.save_results import save_results
from utils.eval import eval_metrics

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

    ### get the dataset names
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    print(datasets)

    ### get the input and target vars
    input_vars = json.load(open(args.data_config + "input_config.json", "r"))
    input_vars = input_vars["input_var"]

    target_vars = json.load(open(args.data_config + "target_config.json", "r"))
    target_vars = target_vars["target_vars"]
    for index, name in enumerate(datasets):

        X_train, X_test, y_train, y_test = load_samples(
            name, input_vars, target_vars[0], args
        )
        # build a naive random model that predicts the most frequent class of the target
        most_frequent_class = y_train.mode()[0]
        start = time.time()
        y_pred = [most_frequent_class] * len(y_test)
        end = time.time()
        # evaluate the naive model
        walltime = end - start
        metrics = eval_metrics(y_test, y_pred)
        
        filename = (
            "naive"
            + "_"
            + name
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        save_results(filename, "naive", name, metrics, args, walltime, None)
        
        # build a second naive model that predicts random values within the range of the target using numpy
        start = time.time()
        y_pred_random = np.random.randint(y_train.min(), y_train.max(), len(y_test))
        end = time.time()
        walltime = end - start
        metrics_random = eval_metrics(y_test, y_pred_random)
        
        filename = (
            "naive_random"
            + "_"
            + name
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        
        save_results(filename, "naive_random", name, metrics_random, args, walltime, None)
        
        
    # run_eval(MODELS, args, tuning="grid")