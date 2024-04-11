import time

# import pandas as pd
import cudf
import dask.array as da
import dask.dataframe as dd
import dask_ml.feature_extraction.text
import optuna
import sklearn.datasets
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_ml.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

bunch = sklearn.datasets.fetch_20newsgroups()
# df = dd.from_pandas(
#     pd.DataFrame({"text": bunch.data, "target": bunch.target}),
#     npartitions=25,
# )
vect = dask_ml.feature_extraction.text.HashingVectorizer()
# X = vect.fit_transform(df["text"])
# X.compute_chunk_sizes()

# # Format classification labels
# y = df["target"].to_dask_array()

# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=21
# )
df = cudf.DataFrame({"text": bunch.data, "target": bunch.target})
X = vect.fit_transform(df.text.values_host)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=21, shuffle=False
)


def tuner(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 7, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0),
        "gamma": trial.suggest_float("gamma", 0.7, 1.0, step=0.1),
    }
    model = XGBClassifier(
        **params,
        n_jobs=-1,
        objective="multi:softmax",
        # tree_method="gpu_hist",
        tree_method="hist",
        # device="cuda",
        # gpu_id=0,
    )
    model.fit(
        x_train,
        y_train,
        early_stopping_rounds=500,
        eval_set=[(x_test, y_test)],
        verbose=3,
    )
    y_hat = model.predict(x_test)
    return f1_score(y_test, y_hat, average="weighted")


start_time = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(tuner, n_trials=100)
time_elapsed = time.time() - start_time
print(f"Time elapsed: {time_elapsed}")
