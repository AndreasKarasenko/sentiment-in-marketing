# load libraries
import matplotlib.pyplot as plt
from utils.dataloader import googleplay
from utils.describe import describe
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from setfit import sample_dataset, SetFitModel, TrainingArguments, Trainer

# load the data
df = googleplay(config=None, path="./data/ikea_reviews.csv")
df = df.iloc[:1000,:] # subsample for faster processing
print(describe(df, "content", score_col="score"))


X = df["content"]
y = df["score"]



# load the model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocess the data
vectorizer = TfidfVectorizer()
encoder = LabelEncoder()

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# fit the model
model.fit(x_train, y_train)

# calculate the sentiment
preds = model.predict(x_test)

# evaluate the performance
print(classification_report(y_test, preds))