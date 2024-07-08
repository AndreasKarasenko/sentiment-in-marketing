### script to run the multivariate analysis
import json
import argparse
from config.utils_config.argparse_args import arguments
from utils.dataloader import get_config_names
from utils.multivariate_analysis import get_statistics


import pandas as pd

parser = argparse.ArgumentParser(
    description="Runs multivariate analysis"
)
for arg in arguments:
    parser.add_argument(
        arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
    )

args = parser.parse_args()


results_all = pd.read_csv("results/overview/summary_all_F1.csv")
results_all.rename(columns={"Unnamed: 0": "model"}, inplace=True)
results_all.dropna(inplace=True)

dataset_statistics = get_statistics(args)