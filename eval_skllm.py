
import argparse
import json
import os
import time
import torch
import gc

# import utilities
from datetime import datetime

# typing

import pandas as pd
import numpy as np

### import the configurations
from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.eval import eval_metrics
from utils.save_results import save_results

import warnings
import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from utils.optimize import run_gridsearchcv
from utils.save_results import save_results

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics
from utils.openai.get_samples import get_samples

### Import the config variables from config/utils_config/argparse_args.py
from config.utils_config.argparse_args import arguments
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier

from skllm.config import SKLLMConfig
from config.utils_config.openai_key import key

# Configure the credentials
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

if __name__ == "__main__":
    
    SKLLMConfig.set_openai_key(key)

    parser = argparse.ArgumentParser(
        description="Finds the optimal Hyperparameters for the models."
    )
    for arg in arguments:
        parser.add_argument(
            arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
        )

    args = parser.parse_args()

    
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    
    model_name = "openai"
    
    
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
    
    
    ### get the input and target vars
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    input_vars = json.load(
        open(args.data_config + "input_config.json", "r")
    )  # for specifying input
    input_vars = input_vars["input_var"]

    target_vars = json.load(
        open(args.data_config + "target_config.json", "r")
    )  # for specifying target
    target_vars = target_vars["target_vars"]

    for i in target_vars:
        samples = get_samples(train_df, n_samples=4, label_col=i)

        clf = FewShotGPTClassifier().fit(
            samples[input_vars].tolist(), samples[i].tolist()
        )
        # clf = DynamicFewShotGPTClassifier(model="gpt-3.5-turbo")
        # clf.fit()

        X_test = test_df[input_vars].tolist()
        y_test = test_df[i].tolist()

        preds = clf.predict(
            X_test,
        )
        np.save("preds_" + i + ".npy", preds)
        metrics = eval_metrics(y_test, preds)

        filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_results(filename, model_name, i, metrics, args)