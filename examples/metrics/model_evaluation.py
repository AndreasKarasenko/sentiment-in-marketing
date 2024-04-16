from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import ComplementNB

# load your model
model = ComplementNB()
# load your data, in this example fetch 20 newsgroups
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
X = twenty_train.data
y = twenty_train.target
twenty_test = fetch_20newsgroups(subset="test", shuffle=True, random_state=42)
X_test = twenty_test.data
y_test = twenty_test.target


# create predictions
preds = model.predict(X)
