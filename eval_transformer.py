from random import shuffle
import warnings
import argparse
import os
import json
import numpy as np
from optree import tree_sum
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from datetime import datetime
from utils.save_results import save_results
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification,
                          DataCollatorWithPadding, create_optimizer)
from typing import List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics

mb = 12282
gpu = tf.config.list_physical_devices("GPU")[0]
tf.config.set_logical_device_configuration(gpu,
                                           [
                                               tf.config.LogicalDeviceConfiguration(memory_limit= mb - 400)
                                               ]
                                           )


import time
# import utilities
from typing import Any, Callable, Dict, List

### import the configurations
from config.utils_config.argparse_args import arguments

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

for name in checkpoints.keys():
    
    model_name = name
    checkpoint = checkpoints[model_name]

    ### get dataset names
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    print(datasets)
    
    ### get the input and target vars
    input_vars = json.load(open(args.data_config + "input_config.json", "r"))
    input_vars = input_vars["input_var"]
    
    target_vars = json.load(open(args.data_config + "target_config.json", "r"))
    target_vars = target_vars["target_vars"]
    
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv") # type: ignore
        test = pd.read_csv(args.data_dir + i + "_test.csv") # type: ignore
        
        train = train.loc[:, [input_vars, target_vars[0]]]
        test = test.loc[:, [input_vars, target_vars[0]]]
        
        train.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        test.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        
        train.label -= 1
        test.label -= 1
        
        test_copy = test.copy()
        
        print(train.head())
        
        path = "results/train/" + model_name + "/" + i
        if not os.path.exists(path):
            os.makedirs(path)
            
        model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=train.label.nunique(), ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)
        
        train = Dataset.from_pandas(train)
        train = train.map(preprocess_function, batched=True)
        
        test = Dataset.from_pandas(test)
        test = test.map(preprocess_function, batched=True)
        
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
        
        bs = 3
        
        train = train.to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=True,
            batch_size=bs,
            collate_fn=collator
        )
        
        test = test.to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=False, # don't shuffle or we can't evaluate!
            batch_size=bs,
            collate_fn=collator
        )
        
        batch_size = bs
        num_epochs = 3
        
        batches_per_epoch = len(train) // batch_size
        total_training_steps = int(batches_per_epoch * num_epochs)
        optimizer, schedule = create_optimizer(init_lr=1e-5, num_warmup_steps=0, num_train_steps=total_training_steps)
        
        model.compile(optimizer=optimizer)
        print(model.summary())
        
        start = time.time()
        model.fit(train, epochs=num_epochs)
        model.save_weights(path + f"/{model_name}")
        
        preds_test = model.predict(test)
        predicted_test = np.argmax(preds_test.logits, axis=1)
        actual_test = test_copy.label
        
        metrics = eval_metrics(actual_test, predicted_test)
        end = time.time()
        walltime = end - start
        print(precision_recall_fscore_support(actual_test, predicted_test, average="weighted"))
        print(metrics[-1])
        
        filename = model_name + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_results(filename, model_name, i, metrics, args, walltime=walltime)