### creates random samples (80/20) from the csv dataset in ./data/ and saves them in ./samples/
### the samples use a stratified split to ensure that the class distribution is the same in both sets
# TODO change this to a function --> should be done in cookiecutter template

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

def split(data: pd.DataFrame, name: str, verbose: bool = False) -> None: 
    """
    Splits the data into a training and test set.
    
    Parameters:
    data (pd.DataFrame): the data to be split
    
    Returns:
    None
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    if verbose:
        print(f"{name}_train: {train.shape}")
        print(f"{name}_test: {test.shape}")
        
    train.to_csv(os.path.join("samples", f"{name}_train.csv"), index=False)
    test.to_csv(os.path.join("samples", f"{name}_test.csv"), index=False)
    
def subsplit(name: str, n_samples: int=1):
    """
    Subsamples the data for each rating for a smaller test set.
    Only use the resulting data for prompt based evals.
    
    Parameters:
    name (str): the name of the dataset
    n_samples (int): the number of samples to take from each rating
    
    Returns:
    """
    path = os.path.join("samples", name)
    data = pd.read_csv(path)
    
    subsample = data.groupby("rating").apply(lambda df: df.sample(n=n_samples)) # for each rating get n_samples
    subsample.to_csv(os.path.join("samples/subsamples", name), index=False)
    