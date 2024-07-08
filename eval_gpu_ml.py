# main script to eval all gpu accelerated non deep learning based models
# SVM, XGBoost
# Since some models benefit more from GPU acceleration than others we only consider the examples where a speedup is expected
# Initial testing for SVM showed a walltime of 13,682s on CPU (excessive)

import argparse
import json
import os
import time

# import utilities
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd

### import the configurations
from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.eval import eval_metrics
from utils.optimize import run_gridsearchcv
from utils.preprocess import TextPreprocessor
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

### import the models dict
from models import GPU_MODELS


def run_eval(
    datasets: List[str],
    model_dict: Dict[str, Callable[[], Any]],
    args: argparse.Namespace,
):
    if args.njobs != 1:
        ErrorString = "njobs must be set to 1 for this script\nIgnoring this can lead to unexpected behavior and freeze your computer"
        raise ValueError(ErrorString)
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv")
        train.dropna(inplace=True)
        test = pd.read_csv(args.data_dir + i + "_test.csv")
        test.dropna(inplace=True)

        X_train = train[input_vars]
        X_test = test[input_vars]
        X_train = TextPreprocessor().fit_transform(X_train)
        X_test = TextPreprocessor().transform(X_test)

        y_train = train[target_vars]
        y_test = test[target_vars]

        # get the model
        for model_name, model_func in model_dict.items():
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
                mode="gpu",
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
                + "gpu_model"
            )
            save_results(filename, model_name, i, metrics, args, walltime, grid)

    return 1


if __name__ == "__main__":
    run_eval(datasets, GPU_MODELS, args)

### notes
# it works but and seems faster than the CPU version but increases the GPU memory usage each time it runs
# there may be a workaround by changing the data loading method to a dask setup and relaunching the client each time
# that may however increase the walltime and introduce overhead
# alternative: https://github.com/rapidsai/cuml/issues/1650 delete the model and clear the memory
