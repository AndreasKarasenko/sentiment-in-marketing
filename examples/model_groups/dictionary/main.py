# load libraries
import pandas as pd
from utils.dataloader import googleplay
from utils.describe import describe

# load the data
df = googleplay(config=None, path="./data/ikea_reviews.csv")
df = df.iloc[:1000,:] # subsample for faster processing
print(describe(df, "content", score_col="score"))

# load the model
from models import DICTIONARY_MODELS

model = DICTIONARY_MODELS["Vader"] # gets the function for the model
model_instance = model()

# calculate the sentiment
df["sentiment"] = df["content"].apply(lambda x: model_instance.polarity_scores(x)["compound"])

# plot sentiment
import matplotlib.pyplot as plt
df["sentiment"].hist()
plt.show()