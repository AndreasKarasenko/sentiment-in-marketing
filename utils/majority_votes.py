### short script to run majority voting on the datasets
### constructs have a suffix (, Y, Z) corresponding to the expert voter
### each vote is a number from 1-5, the majority vote is taken as the final label
### constructs are PE, PI, PEOU, PU, ATT, BI

import pandas as pd
import numpy as np
import os
import json
import argparse
import warnings
from datetime import datetime

# define function
def majority_vote(df, construct: str):
    """Takes a dataframe and a construct and returns a majority vote for that construct"""
    v1 = construct
    v2 = construct + "Y"
    v3 = construct + "Z"
    
    df[construct] = df[[v1, v2, v3]].mode(axis=1)[0]
    return df