import warnings
import argparse
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from datetime import datetime
from utils.save_results import save_results
from utils.openai.get_samples import get_samples
from typing import Any, Callable, Dict, List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from setfit import sample_dataset
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics
import time

from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.dataloader import get_config_names, load_samples



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
    
    model_name = "few_shot_gpt"
    

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
            
        
        samples = get_samples(train, n_samples=8, label_col="label")
        if not os.path.exists("results/train/" + model_name):
            os.makedirs("results/train/" + model_name)
            
        # model = SetFitModel(model_body=SentenceTransformer(checkpoint),
        #                     model_head=LogisticRegression(class_weight="balanced"))
        
        
        start = time.time()
        model = FewShotGPTClassifier(key=API_KEY)
        model = model.fit(samples["text"].tolist(), samples["label"].tolist())
        
        X_test = test["text"].tolist()
        y_test = test["labels"].tolist()
        
        preds = model.predict(test.text.tolist()) # type: ignore
        actual = test.label
        
        metrics = eval_metrics(actual, preds) # type: ignore
        end = time.time()
        walltime = end - start
        print(metrics[-1])
        print(walltime)
        
        filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_results(filename, model_name, i, metrics, args, walltime=walltime)