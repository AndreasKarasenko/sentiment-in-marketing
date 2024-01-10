### creates random samples (80/20) from the csv dataset in ./data/ and saves them in ./samples/
### the samples use a stratified split to ensure that the class distribution is the same in both sets
### the relevant variables for stratifying are: score,PI,PE,PU,PEOU,ATT

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(0)

# Read in data
df = pd.read_csv(os.path.join("data", "data.csv"))

# Create stratified train and test sets

# Create stratified train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test sets using module paths
train.to_csv(os.path.join("samples", "train.csv"), index=False)
test.to_csv(os.path.join("samples", "test.csv"), index=False)


# Print the shape of train and test
print("Train shape:", train.shape)
print("Test shape:", test.shape)
