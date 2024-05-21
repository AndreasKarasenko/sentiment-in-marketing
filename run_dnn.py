### a script that takes json config files from ./config/ and uses gridsearchcv to find the best hyperparameters for each model and saves the optimal parameters to ./config/optimized
# Path: utils/optimize.py
# Import necessary libraries
import time
import argparse
import json
import pandas as pd
from datetime import datetime
from utils.optimize import run_gridsearchcv
from utils.save_results import save_results
import numpy as np

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics

### Import the config variables from config/utils_config/argparse_args.py
from config.utils_config.argparse_args import arguments
# from models import MODELS


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


    ### get the dataset names
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    print(datasets)
    
    ### load the search_space json file
    search_space = json.load(open(args.config_dir, "r"))
    

    ### get the input and target vars
    input_vars = json.load(open(args.data_config + "input_config.json", "r"))
    input_vars = input_vars["input_var"]

    target_vars = json.load(open(args.data_config + "target_config.json", "r"))
    target_vars = target_vars["target_vars"]

    print("input_vars", input_vars)
    print("target_vars", target_vars)
    
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv") # type: ignore
        test = pd.read_csv(args.data_dir + "/subsamples/" + i + "_test.csv") # type: ignore
        
        train = train.loc[:, [input_vars, target_vars[0]]]
        test = test.loc[:, [input_vars, target_vars[0]]]
        
        train.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        test.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        
        # assert that the labels start at 0
        if train.label.min() == 1:
            train.label -= 1
            train.label = train.label.astype(int)
            test.label -= 1
            test.label = test.label.astype(int)
        
        X_train = train["text"]
        X_test = test["text"]
        
        y_train = train["label"]
        y_test = test["label"]
        
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        # y_train = np.asarray(train.label)
        # y_test = np.asarray(test.label)
        
        print(y_train.shape, X_train.shape, y_test.shape, X_test.shape)
        
        from models import DNN_MODELS
        
        for model_name in DNN_MODELS.keys():
            model_func = DNN_MODELS[model_name]
            model_instance = model_func
            start = time.time()
            grid, metrics = run_gridsearchcv(
                model_instance,
                search_space[model_name]["hyperparameters"],
                X_train,
                y_train,
                X_test,
                y_test,
                verbose=args.verbose,
            )
            end = time.time()
            walltime = end - start
            print(metrics[-1])
            
            filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_results(filename, model_name, i, metrics, args, walltime=walltime)