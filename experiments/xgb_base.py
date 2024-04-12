# A quick example for running xgb on the GPU
# This includes offloading data to the GPU, and running the model on the GPU
import time

import cupy as cp
import xgboost as xgb
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score

cluster = LocalCUDACluster()
client = Client(cluster)
cv = HashingVectorizer()

# Load corpus
twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
y = cp.asarray(twenty_train.target).astype(cp.int32)

Xdm = xgb.DMatrix(xformed, label=y)
# model = XGBClassifier(device="cuda", booster="gbtree")
start = time.time()
params = {
    "device": "cuda",
    "booster": "gbtree",
    "eval_metric": "auc",  # figure out custom scoring https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
    "objective": "multi:softprob",
    "num_class": len(cp.unique(y)),
}
# model = xgb.XGBClassifier(
#     device="cuda",
#     booster="gbtree",
#     eval_metric="auc",
#     objective="multi:softprob",
#     num_class=len(cp.unique(y)),
# )
model = xgb.train(params, Xdm)
# model.fit(xformed, y)
end = time.time()
print("Time to train: ", end - start)
print(model.eval(Xdm))
print(f1_score(y.get(), model.predict(Xdm).argmax(axis=1), average="weighted"))
