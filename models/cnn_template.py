import tensorflow_addons as tfa
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn.base import TransformerMixin, BaseEstimator
from keras import regularizers

# use this as a guide https://stackoverflow.com/questions/71288313/using-kerasclassifier-for-training-neural-network
# adding a bert embedding before the other models might not be sensible https://stackoverflow.com/questions/71710186/creating-word-embedings-from-bert-and-feeding-them-to-random-forest-for-classifi
# however glove and other pretrained embeddings exist (word2vec, fasttext, ELMo, BERT, GPT, Universal Sentence Encoder, etc.) 
# these can be used with e.g. cnn https://www.kaggle.com/code/poigal/cnn-on-glove-word-embedding


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, max_features=5000, max_length=200):
        self.tokenizer = Tokenizer(num_words=max_features)
        self.max_length = max_length

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self 

    def transform(self, X, y=None):
        sequences = self.tokenizer.texts_to_sequences(X)
        return pad_sequences(sequences, maxlen=self.max_length)

def build_keras_cnn(hidden_layer_dim, vocab_size, sequence_length, num_classes, activation='relu'):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=sequence_length, mask_zero=True))
    for dim in hidden_layer_dim:
        model.add(Conv1D(dim, 5, activation=activation, padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
    
    
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='weighted')])
    return model 