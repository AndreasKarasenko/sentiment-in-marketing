### functions for multivariate analysis of model performance
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from utils.dataloader import load_samples, get_config_names

def identify_netspeak(text: str):
    """
    Count the number of netspeak usages
    
    Parameters
    ----------
    text : str
        Text containing multiple words
        
    Returns
    -------
    int
        Count of how often netspeak was used
    
    """
    netspeak_words = ['lol', '4ever', 'brb', 'btw', 'bff', 'omg', 'ttyl', 'idk', 'imo', 'lmk', 'np', 'plz', 'thx', 'yolo', 'afaik', 'fomo', 'smh', 'tbh']
    return np.sum([word in netspeak_words for word in text.split()])


def get_statistics(args: argparse.Namespace):
    """
    Calculates varios statistics for datasets
    
    Parameters
    ----------
    datasets : List[str]
        list of datasets from the config file
        
    return : None
    """
    datasets, input_vars, target_vars = get_config_names(args)
    statistics = {}
    for name in tqdm(datasets, desc="Building summary statistics"):
        X_train, _, y_train, _ = load_samples(name, input_vars, target_vars, args)
        ## stats for y
        num_classes = int(y_train.nunique())
        
        ## stats for X
        avg_words_per_text = np.average([len(i.split()) for i in X_train])
        avg_letter_per_word = np.average([len(i) for i in " ".join(X_train).split()])
        max_letters = np.max([len(i) for i in " ".join(X_train).split()])
        # which word? " ".join(train).split()[np.argmax([len(i) for i in " ".join(X_train).split()])]
        # netspeak usage
        avg_netspeak = np.average([identify_netspeak(i) for i in X_train])
        abs_netspeak = np.sum([identify_netspeak(i) for i in X_train])
        
        ## dataset stats
        num_samples = len(X_train)
        
        statistics["avg_words"] = avg_words_per_text
        statistics["avg_letters"] = avg_letter_per_word
        statistics["num_classes"] = num_classes
        statistics["max_letters"] = max_letters
        statistics["avg_netspeak"] = avg_netspeak
        statistics["abs_netspeak"] = abs_netspeak
        statistics["num_samples"] = num_samples
        
    return statistics

def run_multivariate(statistics: dict):
    return