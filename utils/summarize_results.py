import argparse
import datetime
import json
import os
import re
from datetime import datetime
from unittest import result

import numpy as np
import pandas as pd

import data


def summarize_results(mode: str = "all", metric: str = "F1", verbose: bool = False):
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
    datetime_regex_str = "\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"  # regular expression for our datetime format
    dataset_regex_str = (
        "_([^_]+.*?)(?=_\d{4}-\d{2}-\d{2})"  # regular expression for our dataset format
    )

    models = os.listdir("results/train/")  # all model dirs
    models.remove("__init__.py")  # except for __init__.py

    all_results = []
    all_times = []
    mean_times = []
    for model in models:
        files = []
        model_dir = os.path.join("results/train", model)
        # all json files for a model
        # os.listdir lists all files in the model directory
        model_results = [
            file for file in os.listdir(model_dir) if file.endswith(".json")
        ]
        model_results_datasets = set(
            [re.search(dataset_regex_str, file).group(1) for file in model_results]
        )  # all unique datasets
        for dataset in model_results_datasets:
            dataset_results = [file for file in model_results if dataset in file]
            latest_dataset = np.argmax(
                [
                    datetime.strptime(
                        re.search(datetime_regex_str, file).group(0),
                        "%Y-%m-%d_%H-%M-%S",
                    )
                    for file in dataset_results
                ]
            )
            # get the latest dataset results
            files.append(dataset_results[latest_dataset])

        data = [json.load(open(os.path.join(model_dir, file))) for file in files]
        metrics = {
            d["dataset"]: d["metrics"][:-1] for d in data
        }  # exclude the last metric which is the classification report
        time = {d["dataset"]: d["walltime"] for d in data}
        df = pd.DataFrame(metrics)
        df.index = ["ACC", "F1", "PREC", "REC"]
        df_time = pd.DataFrame(time, index=[0])
        if mode == "average":
            df = df.mean(axis=1)
            all_results.append({model: df.apply(lambda x: round(x * 100, 2))})
            all_times.append({model: df_time.sum(axis=1)})
            mean_times.append({model: df_time.mean(axis=1)})
            # print(all_times)
        elif mode == "all":
            df = df.apply(lambda x: x.apply(lambda y: round(y * 100, 2)))
            all_results.append({model: df})
            all_times.append({model: df_time})
        else:
            raise ValueError("mode must be either 'average' or 'all'")
    # print(type(all_results))
    if mode == "average":
        df = pd.DataFrame()
        for i in all_results:
            df = pd.concat([df, pd.DataFrame(i)], axis=1)
        df_time = pd.DataFrame()
        for i in all_times:
            df_time = pd.concat([df_time, pd.DataFrame(i)], axis=1)
        df_mean = pd.DataFrame()
        for i in mean_times:
            df_mean = pd.concat([df_mean, pd.DataFrame(i)], axis=1)
        dfMean = df_mean.T
        dfMean.rename(columns={0: "mean_time"}, inplace=True)
        dfTime = df_time.T
        dfTime.rename(columns={0: "walltime"}, inplace=True)
        dfT = df.T
        dfT = pd.concat([dfT, dfTime, dfMean], axis=1)
        dfT.to_csv("results/overview/summary_average.csv")
        dfT.to_excel("results/overview/summary_average.xlsx")
    elif mode == "all":
        df = pd.DataFrame()
        for i in all_results:
            key, value = list(i.items())[0]
            relevant_value = value.loc[metric, :].to_frame(name=key)
            df = pd.concat([df, pd.DataFrame(relevant_value)], axis=1)
        dfT = df.T
        dfT.to_csv("results/overview/summary_all_" + metric + ".csv")
        dfT.to_excel("results/overview/summary_all_" + metric + ".xlsx")
        dfTime = pd.DataFrame()
        for j in all_times:
            key, value = list(j.items())[0]
            relevant_value = value.loc[0, :].to_frame(name=key)
            dfTime = pd.concat([dfTime, pd.DataFrame(relevant_value)], axis=1)
        dfTime = dfTime.T
        dfTime.to_csv("results/overview/summary_all_walltime.csv")
        dfTime.to_excel("results/overview/summary_all_walltime.xlsx")
    if verbose:
        print(dfT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize results based on models and datasets."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Can either average over all datasets (average) for a model or show results per model per dataset (all).",
    )
    parser.add_argument(
        "--metric", type=str, default="F1", help="Metric to summarize results on."
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Print the result to console"
    )
    args = parser.parse_args()
    summarize_results(mode=args.mode, metric=args.metric, verbose=args.verbose)
