import time

import cupy as cp
import dask
import dask.array as da
import xgboost as xgb

# from xgboost import XGBClassifier
from cuml.dask.common import to_sparse_dask_array
from cuml.dask.naive_bayes import MultinomialNB as cuNB
from cuml.ensemble import RandomForestClassifier as cuRF
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
# from dask_ml.xgboost import XGBClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as skNB

# Create a local CUDA cluster
cluster = LocalCUDACluster()
client = Client(cluster)

# Load corpus
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
cv = CountVectorizer()
xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
X = to_sparse_dask_array(xformed, client)
y = dask.array.from_array(twenty_train.target, asarray=True, fancy=False).astype(
    cp.int32
)
# Train model
model = cuNB()
start = time.time()
model.fit(X, y)
end = time.time()
print("Time to train: ", end - start)

model = skNB()
start = time.time()
model.fit(X, y)
end = time.time()
print("Time to train: ", end - start)

Xda = da.random.random((1000, 10), chunks=100)
yda = da.random.random((1000), chunks=100)

print(type(X))
print(type(Xda))

regressor = xgb.dask.DaskXGBClassifier(
    n_estimators=100, tree_method="gpu_hist", random_state=42
)
regressor.client = client
regressor.set_params(tree_method="gpu_hist", device="cuda")
regressor.fit(X, y)

# Compute accuracy on training set
print(model.score(X, y))
# client.close()
# cluster.close()
