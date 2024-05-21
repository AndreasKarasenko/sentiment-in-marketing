from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from keras.backend import clear_session
from scikeras.wrappers import KerasClassifier
from keras.layers import Embedding

def get_cnn_model(meta, hidden_layer_dim, activation):
    clear_session()
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    vocab_size = meta["vocab_size"]
    embedding_dim = meta["embedding_dim"]
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=n_features_in_))
    model.add(Conv1D(32, 3, activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    
    for dim in hidden_layer_dim:
        model.add(Conv1D(dim, 3, activation=activation))
        model.add(MaxPooling1D(pool_size=2))
        
    model.add(Flatten())
    model.add(Dense(n_classes_, activation='softmax'))
    return model

clf = KerasClassifier(
    model=get_cnn_model,
    hidden_layer_dim=(32,),
    activation="relu",
    loss="categorical_crossentropy",
    verbose=2
)