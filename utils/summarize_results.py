import datetime
import os
import re
import json
import argparse
from unittest import result
import pandas as pd
import numpy as np
from datetime import datetime

import data


def summarize_results(mode: str = "all", metric: str = "f1"):
    """
    Summarize results in ./results/train/ based on models and datasets.
    Can either average over all datasets (average) for a model or show results per model per dataset (all).

    Parameters
    ----------
    mode : str
        "average" or "all"
        default: "model"
        
    return : None
    """
    datetime_regex_str = "\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}" # regular expression for our datetime format
    dataset_regex_str = "_([^_]+.*?)(?=_\d{4}-\d{2}-\d{2})" # regular expression for our dataset format
    
    models = os.listdir("results/train/") # all model dirs
    models.remove("__init__.py") # except for __init__.py
    
    all_results = []
    for model in models:
        files = []
        model_dir = os.path.join("results/train", model)
        model_results = [file for file in os.listdir(model_dir) if file.endswith(".json") ]
        model_results_datasets = set([re.search(dataset_regex_str, file).group(1) for file in model_results]) # all unique datasets
        for dataset in model_results_datasets:
            dataset_results = [file for file in model_results if dataset in file]
            latest_dataset = np.argmax([datetime.strptime(re.search(datetime_regex_str, file).group(0), "%Y-%m-%d_%H-%M-%S") for file in dataset_results])
            files.append(dataset_results[latest_dataset])
            
        data = [json.load(open(os.path.join(model_dir, file))) for file in files]
        metrics = {d["dataset"]: d["metrics"][:-1] for d in data} # exclude the last metric which is the classification report
        df = pd.DataFrame(metrics)
        df.index = ["ACC", "F1", "PREC", "REC"]
        if mode == "average":
            all_results.append({model:df.mean(axis=1)})
        elif mode == "all":
            all_results.append({model:df})
        else:
            raise ValueError("mode must be either 'average' or 'all'")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize results based on models and datasets.')
    parser.add_argument('--mode', type=str, default='all', help='Can either average over all datasets (average) for a model or show results per model per dataset (all).')
    parser.add_argument('--metric', type=str, default='f1', help='Metric to summarize results on.')
    args = parser.parse_args()
    print(summarize_results(mode = args.mode, metric = args.metric))
    
# models = os.listdir("results/train/") # all model dirs
# models.remove("__init__.py") # except for __init__.py

# all_results = []
# for model in models:
#     model_dir = os.path.join("results/train", model)
#     model_results = [file for file in os.listdir(model_dir) if file.endswith(".json") ]
#     print(model_results)



# datetime_regex_str = "\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}" # regular expression for our datetime format
# dataset_regex_str = "_([^_]+.*?)(?=_\d{4}-\d{2}-\d{2})" # regular expression for our dataset format
# model_results_datasets = [re.search(dataset_regex_str, file).group(1) for file in model_results]
# print(model_results_datasets)
# for dataset in model_results_datasets:
#     print(dataset)
#     dataset_results = [file for file in model_results if dataset in file]
#     print(dataset_results)
    
    
# comp = ['MultinomialNB_patio_lawn_garden_2024-04-02_14-11-39.json', 'MultinomialNB_patio_lawn_garden_2024-04-03_11-56-40.json']
# dates = [datetime.strptime(re.search(datetime_regex_str, file).group(0), "%Y-%m-%d_%H-%M-%S") for file in comp]
# ['2024-04-02_14-11-39', '2024-04-03_11-56-40']