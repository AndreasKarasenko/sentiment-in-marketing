import time

import cudf
import cupy as cp
import numpy as np

# from xgboost import XGBClassifier
from cuml.dask.common import to_sparse_dask_array
from cuml.ensemble import RandomForestClassifier

# from dask_ml.feature_extraction.text import HashingVectorizer
from cuml.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

# from cuml.dask.naive_bayes import MultinomialNB as cuNB
from cuml.naive_bayes import MultinomialNB as cuNB
from cuml.svm import SVC as cuSVC
from cupyx.scipy.sparse import csr_matrix
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.datasets import fetch_20newsgroups

# Create a local CUDA cluster
cluster = LocalCUDACluster()
client = Client(cluster)

# Load corpus

twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
twenty_train = cudf.DataFrame.from_dict(
    {"data": twenty_train.data, "target": twenty_train.target}
)
cv = HashingVectorizer()

xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)

X = csr_matrix(xformed).astype(cp.float32)
y = cp.asarray(twenty_train.target).astype(cp.int32)

from cuml.ensemble import RandomForestClassifier as cuRF

# Try NB
model = cuNB()
start = time.time()
model.fit(X, y)
end = time.time()
print("Time to train: ", end - start)

# Try SVC
model = cuSVC(
    kernel="linear", gamma="scale"
)  # use hashing vectorizer and its super fast 16s TFIDF takes 15s approx
start = time.time()
model.fit(X, y)
end = time.time()
print("Time to train: ", end - start)
model.score(X, y)  # tfidf: 0.9618, hashing: 0.9618

# Try RF
X_dense = cp.asarray(X.todense())
model = cuRF()
start = time.time()
model.fit(X_dense, y)  # works but takes 30s and only with CountVectorizer
end = time.time()
print("Time to train: ", end - start)

from sklearn.feature_extraction.text import CountVectorizer as skCV
from sklearn.feature_extraction.text import HashingVectorizer as skHV
from sklearn.feature_extraction.text import TfidfVectorizer as skTV

# Try RF CPU
# Change the data to a sklearn supported format
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
X = skTV().fit_transform(twenty_train.data)
y = twenty_train.target

from sklearn.ensemble import RandomForestClassifier as skRF

model = skRF()
start = time.time()
model.fit(
    X, y
)  # takes 28s this also coincides with experiments with random_forest_example.py
# skHV takes 172s
# tfidf takes 162s
end = time.time()
print("Time to train: ", end - start)
