import numpy as np
from utils.dataloader import googleplay
from utils.describe import describe
from datasets import Dataset
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification,
                          DataCollatorWithPadding, create_optimizer)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load the data
df = googleplay(config=None, path="./data/ikea_reviews.csv")
df = df.iloc[:1000,:] # subsample for faster processing
df.rename(columns={"content": "text", "score": "label"}, inplace=True)
print(describe(df, "text", score_col="label"))


# number of unique score labels
n_labels = df["label"].nunique()

# load the model
checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=n_labels, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

encoder = LabelEncoder()
train = Dataset.from_pandas(train)
train = train.map(preprocess_function, batched=True)

test = Dataset.from_pandas(test)
test = test.map(preprocess_function, batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

bs = 3
num_epochs = 3

train = train.to_tf_dataset(columns=["input_ids", "attention_mask", "label"], batch_size=bs, collate_fn=collator)
test = test.to_tf_dataset(columns=["input_ids", "attention_mask", "label"], batch_size=bs, collate_fn=collator)

batches_per_epoch = len(train) // bs
total_training_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=1e-5, num_warmup_steps=0, num_train_steps=total_training_steps)

model.compile(optimizer=optimizer)
print(model.summary())
# fit the model
model.fit(train, epochs=num_epochs)

# calculate the sentiment
preds = model.predict(test)
preds = np.argmax(preds.logits, axis=1)

# evaluate the performance
print(classification_report(test["label"], preds))