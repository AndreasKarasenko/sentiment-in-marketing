### a script that takes json config files from ./config/ and uses gridsearchcv to find the best hyperparameters for each model and saves the optimal parameters to ./config/optimized
# Path: utils/optimize.py
# Import necessary libraries
import argparse
import json
from weakref import ref
import numpy as np
import pandas as pd
from datetime import datetime
from utils.save_results import save_results
import time
import gc

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from models.cnn_template import build_keras_cnn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

### Import evaluation functions from utils/eval.py
from utils.eval import eval_metrics

### Import the config variables from config/utils_config/argparse_args.py
from config.utils_config.argparse_args import arguments

# from models import MODELS

from scikeras.wrappers import KerasClassifier

if __name__ == "__main__":
    print("test")
    parser = argparse.ArgumentParser(
        description="Finds the best hyperparameters for the models"
    )
    for arg in arguments:
        parser.add_argument(
            arg["arg"], type=arg["type"], default=arg["default"], help=arg["help"]
        )

    args = parser.parse_args()


    ### get the dataset names
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]
    print(datasets)
    
    ### load the search_space json file
    search_space = json.load(open(args.config_dir, "r"))
    

    ### get the input and target vars
    input_vars = json.load(open(args.data_config + "input_config.json", "r"))
    input_vars = input_vars["input_var"]

    target_vars = json.load(open(args.data_config + "target_config.json", "r"))
    target_vars = target_vars["target_vars"]

    print("input_vars", input_vars)
    print("target_vars", target_vars)
    
    for i in datasets:
        train = pd.read_csv(args.data_dir + i + "_train.csv") # type: ignore
        test = pd.read_csv(args.data_dir + "/subsamples/" + i + "_test.csv") # type: ignore
        
        train = train.loc[:, [input_vars, target_vars[0]]]
        test = test.loc[:, [input_vars, target_vars[0]]]
        
        train.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        test.rename(columns={input_vars: "text", target_vars[0]: "label"}, inplace=True)
        
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        
        tokenizer = Tokenizer(num_words=30000, lower=True)
        tokenizer.fit_on_texts(train.text)
        word_index = tokenizer.word_index
        
        sequences_train = tokenizer.texts_to_sequences(train.text.values)
        sequences_test = tokenizer.texts_to_sequences(test.text.values)
        
        X_train = pad_sequences(sequences_train, maxlen=200)
        X_test = pad_sequences(sequences_test, maxlen=200)
        
        EMB_DIM = 200
        vocab_size = min(len(word_index) + 1, 30000)
        
        # assert that the labels start at 0
        if train.label.min() == 1:
            train.label -= 1
            train.label = train.label.astype(int)
            test.label -= 1
            test.label = test.label.astype(int)
            
        y_train = np.asarray(train.label)
        y_test = np.asarray(test.label)
        
        print(y_train.shape, X_train.shape, y_test.shape, X_test.shape)
        clf = KerasClassifier(
            build_fn=build_keras_cnn,
            vocab_size=vocab_size,
            sequence_length=200,
            num_classes=len(train.label.unique()),
            epochs=50,
            hidden_layer_dim=(32,),
            activation="relu",
            verbose=1
        )
        
        pipeline = Pipeline([("clf", clf)])
        scoring = {
            "Accuracy": "accuracy",
            "Precision": "precision_weighted",
            "Recall": "recall_weighted",
            "F1": "f1_weighted",
        }
        grid_search = GridSearchCV(
            pipeline,
            search_space["CNN"]["hyperparameters"],
            cv=10,
            verbose=args.verbose,
            scoring=scoring,
            refit="F1",  # refit the model on the best F1 score
            return_train_score=True,
        )
        # model_instance = build_keras_cnn(vocab_size=vocab_size, sequence_length=200, num_classes=5)

        # clf.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        start = time.time()
        grid_search.fit(X_train, y_train)

        predictions = grid_search.predict(X_test)

        metrics = eval_metrics(y_test, predictions)
        end = time.time()
        walltime = end - start
        print(metrics)

        filename = "CNN" + "_" + i + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_results(filename, "CNN", i, metrics, args, walltime=walltime)
        
        del clf, grid_search, train, test
        gc.collect()
        