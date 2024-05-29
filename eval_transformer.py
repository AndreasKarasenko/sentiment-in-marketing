import warnings
import argparse
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from datetime import datetime
from utils.save_results import save_results
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification,
                          DataCollatorWithPadding, create_optimizer)
from typing import List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics


import time
# import utilities
from typing import Any, Callable, Dict, List

### import the configurations
from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.dataloader import get_config_names, load_samples

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
        for model_name, model_string in model_dict.items():
            model = TFAutoModelForSequenceClassification(model_string, num_labels=len(np.unique(y_train)), ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(model_string)
            
            def preprocess_function(examples):
                return tokenizer(examples["text"], truncation=True)
            
            train = Dataset.from_pandas
            
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


def weighted_continous(probs: List[float]):
    score = 0
    probability = 1e-10 # assure that no zero division can occur
    for idx, value in enumerate(probs):
        score += idx * value
        probability += value
    return score / probability


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
parser = argparse.ArgumentParser(
    description="Finds the optimal Hyperparameters for the models."
)
for arg in arguments:
    parser.add_argument(
        arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
    )

args = parser.parse_args()


### load the search_space json file
search_space = json.load(open(args.config_dir, "r"))

### load the checkpoint
checkpoints = json.load(open("./config/model_config/checkpoints.json", "r"))
print(checkpoints)
model_name = "germanbert"
checkpoint = checkpoints[model_name]

### load the data
train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

### split the data into features and labels
# TODO - remove redundant code
input_vars = json.load(open(args.data_config + "input_config.json", "r"))  # for specifying input
input_vars = input_vars["input_var"]

target_vars = json.load(open(args.data_config + "target_config.json", "r")) # for specifying target
target_vars = target_vars["target_vars"]

for i in target_vars:
    train_df = majority_vote(train_df, i)
    train_df[i] = train_df[i] -1
    train = train_df.loc[:,[input_vars, i]]
    train.rename(columns={input_vars:"text",
                          i:"label"}, inplace=True)
    
    test_df = majority_vote(test_df, i)
    test_df[i] = test_df[i] -1
    test = test_df.loc[:,[input_vars, i]]
    test.rename(columns={input_vars:"text",
                          i:"label"}, inplace=True)
    
    if not os.path.exists("results/train/" + model_name):
        os.makedirs("results/train/" + model_name)
        
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if model_name + "_" + i + ".index" in os.listdir("results/train/" + model_name):
        model.load_weights("results/train/" + model_name + "/" + model_name + "_" + i)
        
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    train = Dataset.from_pandas(train)
    train = train.map(preprocess_function, batched=True)
    
    test = Dataset.from_pandas(test)
    test = test.map(preprocess_function, batched=True)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    
    train = train.to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=4,
        collate_fn=collator
    )
    
    test = test.to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=False,
        batch_size=4,
        collate_fn=collator
    )
    
    batch_size = 4
    num_epochs = 3
    
    batches_per_epoch = len(train) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=1e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    model.compile(optimizer=optimizer)
    print(model.summary())
    
    if model_name + "_" + i + ".index" not in os.listdir("results/train/" + model_name):
        model.fit(train, epochs=3)
        model.save_weights("results/train/" + model_name + "/" + model_name + "_" + i)

        
    preds_test = model.predict(test)
    predicted_test = np.argmax(preds_test.logits, axis=1)
    actual_test = test_df[i].values
    
    metrics = eval_metrics(actual_test, predicted_test)
    print(precision_recall_fscore_support(actual_test, predicted_test, average='weighted'))
    print(metrics[-1])
    print(confusion_matrix(actual_test, predicted_test))
    
    filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_results(filename, model_name, i, metrics, args)