from models import (
    complement_Bayes,
    dt,
    logistic_regression,
    naive_bayes,
    rf,
    svm,
    vader,
    xgboost,
)

DICTIONARY_MODELS = {
    "Vader": vader.model,
}

MODELS = {
    # "MultinomialNB": naive_bayes.model,
    # "ComplementNB": complement_Bayes.model,
    "LR": logistic_regression.model,
    # "SVM": svm.model,
    # "Decision Tree": dt.dt_model,
    # "RF": rf.rf_model,
    # "XGBoost": xgboost.xgboost_model,
}

DNN_MODELS = {}

TRANSFER_MODELS = {}
