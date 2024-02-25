import pandas as pd
from utils.dataloader import googleplay
from models import DICTIONARY_MODELS

# load the data
df = googleplay(config=None, path="./data/ikea_en.csv")
df = df.iloc[:1000,:] # subsample for faster processing

# load the model
model = DICTIONARY_MODELS["Vader"] # gets the function for the model
model_instance = model()

# calculate the sentiment
