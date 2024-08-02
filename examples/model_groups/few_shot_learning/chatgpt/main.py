# load libraries
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier

from skllm.config import SKLLMConfig
from config.utils_config.openai_key import key
from utils.openai.get_samples import get_samples
# set the openai key - should look like this: key = "sk-..."
SKLLMConfig.set_openai_key(key)


import matplotlib.pyplot as plt
from utils.dataloader import googleplay
from utils.describe import describe
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset

# load the data
df = googleplay(config=None, path="./data/ikea_reviews.csv")
df = df.iloc[:1000,:] # subsample for faster processing
print(describe(df, "content", score_col="score"))

# preprocess the data
print(df.columns)
df = df.loc[:,["content", "score"]]
encoder = LabelEncoder()
df["score"] = encoder.fit_transform(df["score"])

# split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)
train_data = get_samples(train, n_samples=4, label_col="score") # get 4 per class


# load the model
model = FewShotGPTClassifier()

# fit the model
model.fit(train_data["content"].tolist(), train_data["score"].tolist())

# calculate the sentiment
preds = model.predict(test["content"].tolist())

# evaluate the performance
print(classification_report(test["score"], preds))