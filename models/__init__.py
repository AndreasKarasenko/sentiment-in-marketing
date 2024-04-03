from models import vader, naive_bayes, logistic_regression, svm, dt, rf, xgboost, complement_Bayes

DICTIONARY_MODELS = {
    "Vader": vader.model,
}

MODELS = {
    # "MultinomialNB": naive_bayes.model,
    # "ComplementNB": complement_Bayes.model,
    "LR": logistic_regression.model,
    "SVM": svm.model,
    "Decision Tree": dt.dt_model(),
    "RF": rf.rf_model(),
    "XGBoost": xgboost.xgboost_model()
}

DNN_MODELS = {}

TRANSFER_MODELS = {}
