import cupy as cp
import dask
from cuml.dask.common import to_sparse_dask_array
from cuml.dask.naive_bayes import MultinomialNB
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Create a local CUDA cluster

cluster = LocalCUDACluster()
client = Client(cluster)

# Load corpus
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
cv = CountVectorizer()
xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
X = to_sparse_dask_array(xformed, client)
y = dask.array.from_array(twenty_train.target, asarray=False, fancy=False).astype(
    cp.int32
)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Compute accuracy on training set
print(model.score(X, y))
client.close()
cluster.close()
