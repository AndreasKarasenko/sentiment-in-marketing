Models should be defined individually as .py files and then imported using the \_\_init__.py file

```python
# put this in your __init__.py
# import implemented models
from models import (cnn, naive_bayes, nn,
                    adaboost, lstm, dt, knn, lr,
                    rf, svm, xgboost)


MODELS = {
    "AdaBoostClassifier": adaboost.adaboost_model,
    "Decision Tree": dt.dt_model,
    "KNN" : knn.knn_model,
    "Logistic Regression" : lr.lr_model,
    "RandomForestClassifier" : rf.rf_model,
    "SVM" : svm.svm_model,
    "XGBClassifier" : xgboost.xgboost_model,
    "MultinomialNB": naive_bayes.naive_bayes_model,
}

DNN_MODELS = {
    "Neural Network": nn.clf,
    "LSTM": lstm.clf,
    "CNN": cnn.clf,
}
```
```python
# then import elsewhere like this or adjust as you like
from models import MODELS, DNN_MODELS
# load and preprocess data
...
# use a specific model
svm = MODELS["svm"]
svm.fit(X_train, y_train)
...
# iterate over all models
results = {}
for model_name in MODELS.keys():
    model_function = MODELS[model_name]
    model_function.fit(train_data)
    results[model_name] = model_function.evaluate(X_test, y_test)
```