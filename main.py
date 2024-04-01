import warnings
import argparse
import os
import json
import pandas as pd
from datetime import datetime
from utils.optimize import run_gridsearchcv
from utils.save_results import save_results

from utils.eval import eval_metrics

from config.utils_config.argparse_args import arguments

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "ignore"
parser = argparse.ArgumentParser(
    description="Finds the optimal Hyperparameters for the models"
)

for arg in arguments:
    parser.add_argument(
        arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
    )
    
args = parser.parse_args()

search_space = json.load(open(args.config_dir, "r"))