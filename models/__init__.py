from models import vader, naive_bayes, logistic_regression, svm

DICTIONARY_MODELS = {
    "Vader": vader.model,
}

MODELS = {
    "MultinomialNB": naive_bayes.model,
    "LR": logistic_regression.model,
    "SVM": svm.model,
}

DNN_MODELS = {}

TRANSFER_MODELS = {}
