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

### Naive Bayes (NB)
NB is a popular algorithm in a variety of classification tasks. Although more modern approaches oftentimes perform better, NB is a simple, fast and inexpensive benchmark. It works by calculating the conditional probability of a class given an attribute set.

If we wish to classify a text B using Naive Bayes, we use the formula:

```markdown
P(A|B) = [P(B|A) * P(A)] / P(B)
```
Here:
- A is the class or category we want to predict (e.g., positive or negative sentiment).
- B is the observed text.
- P(A|B) is the posterior probability of class A given predictor (text).
- P(B|A) is the likelihood which is the probability of predictor given class.
- P(A) is the prior probability of class.
- P(B) is the prior probability of predictor.

Due to its inexpensivenes, we can conduct very thorough HP training.

### Random Forest (RF)

RF are an ensemble learning method and use multiple decision trees that are trained on subsets of the training data. These trees become "experts" and are pooled together for the final prediction. The pooling works by calculating the mode of the classes.

### 