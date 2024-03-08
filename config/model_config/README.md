For the model config we consider checkpoints for pre-trained models from Huggingface and a search_space for hyperparameter tuning.

Both are saved as JSON and should have the following structures.

checkpoints.json
```json
{
  "model_name": "huggingface/link/to/file",
  "bert": "bert-base-multilingual-uncased",
  "siebert": "siebert/sentiment-roberta-large-english",
}
```
search_space.json
```json
{
    "RandomForestClassifier": {
        "hyperparameters": {
            "clf__criterion": ["gini", "entropy"],
            "clf__n_estimators": [10, 20, 50],
            "clf__max_depth": [2, 5, 10, 20, 50, 100],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5]
        }
    },
    "MultinomialNB": {
        "hyperparameters": {
            "clf__alpha": [0.5, 0.4, 0.3, 0.2, 0.1, 1e-1, 1e-2],
            "clf__fit_prior": [true, false],
            "clf__force_alpha": [true]
        }
    }
}
```

the ```clf__``` prefix is a result of the pipeline we use. It simply denotes the classifier and its parameters.
For NB it is NB.alpha or clf__alpha. If we called it "classifier" instead it would be ```classifier__```. The respective hyperparameters can be found in the model documentation (either online or on your local machine).