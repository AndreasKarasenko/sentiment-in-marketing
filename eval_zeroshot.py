# example script to run and evaluate prompt based sentiment analysis using zeroshot

import argparse
import json
import os
import time

# import utilities
import warnings
from datetime import datetime

# typing
from typing import Any, Callable, Dict, List

import pandas as pd

### import the configurations
from config.utils_config.argparse_args import arguments

### import evaluation functions
from utils.dataloader import get_config_names
from utils.eval import eval_metrics
from utils.optimize import run_bayesian_optimization, run_gridsearchcv
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
# TODO change arguments to take the subsamples as test but still use samples for train