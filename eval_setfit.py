# example script to run and evaluate setfit based few shot learning on sentiment data
# unlike zero or few shot prompting setfit first learns a sentence transformer base.
# This base essentially learns to  differentiate sentences that belong to the same class
# from those that belong to different classes.

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

### import the models dict
from setfit import sample_dataset, SetFitModel, TrainingArguments, Trainer
from datasets import Dataset

### import evaluation functions
from utils.eval import eval_metrics
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


    ### get the dataset names
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    print(datasets)
    
    model_name = "setfit"
    checkpoints = json.load(open("./config/model_config/checkpoints.json", "r"))
    checkpoint = checkpoints[model_name]
    

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
        

        # remap labels to 3-class problem
        # data can have 1-3, 1-5, or 1-10 labels so we need to remap to 0-2
        # we do this by assigning the medium label to 1 and the rest to 0 or 2
        train_mean = train.label.unique().mean()
        
        tsmaller = train.label < int(train_mean)
        tlarger = train.label > int(train_mean)
        tequal = train.label == int(train_mean)
        
        train.loc[tsmaller, "label"] = 0
        train.loc[tlarger, "label"] = 2
        train.loc[tequal, "label"] = 1
        
        print(int(test.label.unique().mean()))
        test_mean = test.label.unique().mean()
        
        tsmaller = test.label < int(test_mean)
        tlarger = test.label > int(test_mean)
        tequal = test.label == int(test_mean)
        
        test.loc[tsmaller, "label"] = 0
        test.loc[tlarger, "label"] = 2
        test.loc[tequal, "label"] = 1
        
        # assert that the labels start at 0
        if train.label.min() == 1:
            train.label -= 1
            train.label = train.label.astype(int)
            test.label -= 1
            test.label = test.label.astype(int)
            
        # train.loc[train.label < 2, "label"] = 0
        # test.loc[test.label < 2, "label"] = 0
        # train.loc[train.label > 2, "label"] = 2
        # train.loc[train.label < 3, "label"] = 0
        # test.loc[test.label < 3, "label"] = 0
        # train.loc[train.label >= 3, "label"] = 1
        # test.loc[test.label >= 3, "label"] = 1
        
        train_data = sample_dataset(Dataset.from_pandas(train), label_column="label", num_samples=8)
        if not os.path.exists("results/train/" + model_name):
            os.makedirs("results/train/" + model_name)
            
        # model = SetFitModel(model_body=SentenceTransformer(checkpoint),
        #                     model_head=LogisticRegression(class_weight="balanced"))
        
        model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2",
                                            )
        
        print(torch.cuda.memory_allocated(0))
        arguments = TrainingArguments(
            batch_size=8,
            num_epochs=4,
            evaluation_strategy="epoch",
        )
        arguments.eval_strategy = arguments.evaluation_strategy # type: ignore
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_data,
            metric="accuracy",
            column_mapping={"text": "text", "label": "label"},
        )
        # trainer.train_classifier(train.text, train.label)
        print(args)
        start = time.time()
        trainer.train()
        
        preds = model.predict(test.text) # type: ignore
        actual = test.label
        
        metrics = eval_metrics(actual, preds) # type: ignore
        end = time.time()
        walltime = end - start
        print(metrics[-1])
        print(walltime)
        
        filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_results(filename, model_name, i, metrics, args, walltime=walltime)
        
        del model, trainer, train_data, train, test
        gc.collect()
        torch.cuda.empty_cache()
        