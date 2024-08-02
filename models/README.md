# Sentiment Models
A written overview can be found [here](./MODELS.md), while the hyperparameter specification can be found in the config folder under [search space](../config/model_config/search_space.json). Read also the associated search space [README](../config/model_config/README.md).

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
preds = svm.predict(X_test)
...
```