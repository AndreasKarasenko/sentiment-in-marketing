from tensorflow import keras
from keras.backend import clear_session
from scikeras.wrappers import KerasClassifier


# Neural Networks
def get_model(hidden_layer_dim, activation, meta):
    clear_session()
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation(activation))
    
    for dim in hidden_layer_dim:
        model.add(keras.layers.Dense(dim))
        model.add(keras.layers.Activation(activation))
        
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model

clf = KerasClassifier(
    model=get_model,
    hidden_layer_dim=(32,),
    activation="relu",
    loss="categorical_crossentropy",
    verbose=2
)