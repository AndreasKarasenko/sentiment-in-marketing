# load libraries
import matplotlib.pyplot as plt
from utils.dataloader import googleplay
from utils.describe import describe
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from setfit import sample_dataset, SetFitModel, TrainingArguments, Trainer

from datasets import Dataset

# load the data
df = googleplay(config=None, path="./data/ikea_reviews.csv")
df = df.iloc[:1000,:] # subsample for faster processing
print(describe(df, "content", score_col="score"))

# preprocess the data

df = df.loc[:,["content", "score"]]
encoder = LabelEncoder()
df["score"] = encoder.fit_transform(df["score"])

# split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)
train_data = sample_dataset(Dataset.from_pandas(train), label_column="score", num_samples=4) # get 4 per class


# load the model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

arguments = TrainingArguments(
    batch_size=8,
    num_epochs=4,
    evaluation_strategy="epoch",
)
arguments.eval_strategy = arguments.evaluation_strategy # type: ignore
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_data,
    metric="accuracy",
    column_mapping={"content": "text", "score": "label"},
)

# fit the model
trainer.train()

# calculate the sentiment
preds = model.predict(test["content"]) # type: ignore

# evaluate the performance
print(classification_report(test["score"], preds))