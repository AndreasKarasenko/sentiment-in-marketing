### A function to load the data from data/samples/train.csv and data/samples/test.csv
from typing import List
import pandas as pd
import json
import gzip
import os


def amazon(config: dict, path: str):
    """Load data from amazon review dataset. The dataset is stored as zipped json files."""
    # Load data
    with gzip.open(path, "rb") as f:
        json_data = [json.loads(line) for line in f]
    df = pd.DataFrame.from_dict(json_data)
    
    return df

def googleplay(config: dict, path: str):
    """Load data from google play store review dataset. The dataset is stored as csv files."""
    # Load data
    df = pd.read_csv(path)    
    
    return df