# import time

# import dask.dataframe as dd
# import dask_ml.feature_extraction.text
# import pandas as pd
# import sklearn.datasets
# from dask.distributed import Client
# from dask_cuda import LocalCUDACluster

# # from xgboost.dask import DaskXGBClassifier
# from xgboost import dask as dxgb

# if __name__ == "__main__":  # GPU version

#     with LocalCUDACluster(n_workers=1, threads_per_worker=4) as cluster:
#         with Client(cluster) as client:
#             # Create Dask DataFrame from sklearn 20newsgroups dataset
#             bunch = sklearn.datasets.fetch_20newsgroups()
#             df = dd.from_pandas(
#                 pd.DataFrame({"text": bunch.data, "target": bunch.target}),
#                 npartitions=25,
#             )

#             # Create features with dask-ml's `HashingVectorizer``
#             vect = dask_ml.feature_extraction.text.HashingVectorizer()
#             X = vect.fit_transform(df["text"])

#             # Format classification labels
#             y = df["target"].to_dask_array()
#             dtrain = dxgb.DaskQuantileDMatrix(client, X, y)

#             # Train XGBoost classifier
#             start = time.time()
#             output = dxgb.train(
#                 client,
#                 {
#                     "verbosity": 2,
#                     "tree_method": "hist",
#                     # Golden line for GPU training
#                     "device": "cuda",
#                 },
#                 dtrain,
#                 num_boost_round=4,
#                 evals=[(dtrain, "train")],
#             )
#             end = time.time()
#             print("Time to train: ", end - start)
#             bst = output["booster"]
#             history = output["history"]

import time

import dask.dataframe as dd
import dask_ml.feature_extraction.text
import pandas as pd
import sklearn.datasets
from dask.distributed import Client
from xgboost.dask import DaskXGBClassifier

if __name__ == "__main__":  # CPU version

    with Client():
        # Create Dask DataFrame from sklearn 20newsgroups dataset
        bunch = sklearn.datasets.fetch_20newsgroups()
        df = dd.from_pandas(
            pd.DataFrame({"text": bunch.data, "target": bunch.target}), npartitions=25
        )

        # Create features with dask-ml's `HashingVectorizer``
        vect = dask_ml.feature_extraction.text.HashingVectorizer()
        X = vect.fit_transform(df["text"])

        # Format classification labels
        y = df["target"].to_dask_array()

        # Train XGBoost classifier
        clf = DaskXGBClassifier()
        print(f"{X = }")
        print(f"{y = }")
        start = time.time()
        clf.fit(X, y)  # Results in `AttributeError: divisions not found`
        end = time.time()
        print("Time to train: ", end - start)


# import cupy as cp
# import dask.dataframe as dd
# import dask_cudf
# import dask_ml.feature_extraction.text
# import pandas as pd
# import sklearn.datasets
# from dask import array as da
# from dask import dataframe as dd
# from dask.distributed import Client
# from dask_cuda import LocalCUDACluster
# from xgboost import dask as dxgb
# from xgboost.dask import DaskDMatrix, DaskXGBClassifier


# def using_dask_matrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
#     # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter
#     # around workers.
#     dtrain = DaskDMatrix(client, X, y)

#     # Use train method from xgboost.dask instead of xgboost.  This distributed version
#     # of train returns a dictionary containing the resulting booster and evaluation
#     # history obtained from evaluation metrics.
#     output = dxgb.train(
#         client,
#         {
#             "verbosity": 2,
#             "tree_method": "hist",
#             # Golden line for GPU training
#             "device": "cuda",
#         },
#         dtrain,
#         num_boost_round=4,
#         evals=[(dtrain, "train")],
#     )
#     bst = output["booster"]
#     history = output["history"]

#     # you can pass output directly into `predict` too.
#     prediction = dxgb.predict(client, bst, dtrain)
#     print("Evaluation history:", history)
#     return prediction


# def using_quantile_device_dmatrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
#     """`DaskQuantileDMatrix` is a data type specialized for `hist` tree methods for
#      reducing memory usage.

#     .. versionadded:: 1.2.0

#     """
#     X = dask_cudf.from_dask_dataframe(dd.from_dask_array(X))
#     y = dask_cudf.from_dask_dataframe(dd.from_dask_array(y))

#     # `DaskQuantileDMatrix` is used instead of `DaskDMatrix`, be careful that it can not
#     # be used for anything else other than training unless a reference is specified. See
#     # the `ref` argument of `DaskQuantileDMatrix`.
#     dtrain = dxgb.DaskQuantileDMatrix(client, X, y)
#     output = dxgb.train(
#         client,
#         {"verbosity": 2, "tree_method": "hist", "device": "cuda"},
#         dtrain,
#         num_boost_round=4,
#     )

#     prediction = dxgb.predict(client, output, X)
#     return prediction


# if __name__ == "__main__":
#     # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
#     # `n_workers` represents the number of GPUs since we use one GPU per worker process.
#     with LocalCUDACluster(n_workers=1, threads_per_worker=4) as cluster:
#         with Client(cluster) as client:
#             # generate some random data for demonstration
#             rng = da.random.default_rng(1)

#             bunch = sklearn.datasets.fetch_20newsgroups()
#             df = dd.from_pandas(
#                 pd.DataFrame({"text": bunch.data, "target": bunch.target}),
#                 npartitions=25,
#             )

#             # Create features with dask-ml's `HashingVectorizer``
#             vect = dask_ml.feature_extraction.text.HashingVectorizer()
#             X = vect.fit_transform(df["text"])

#             # Format classification labels
#             y = df["target"].to_dask_array()

#             # m = 100000
#             # n = 100
#             # X = rng.normal(size=(m, n))
#             # y = X.sum(axis=1)

#             print("Using DaskQuantileDMatrix")
#             from_ddqdm = using_quantile_device_dmatrix(client, X, y)
#             print("Using DMatrix")
#             from_dmatrix = using_dask_matrix(client, X, y)
