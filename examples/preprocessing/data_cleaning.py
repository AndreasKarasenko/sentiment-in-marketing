import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin

# load data
X_train, X_test, y_train, y_test = ...


# define a preprocessor
class TextPreprocessor(TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for document in X:
            # Lowercase
            document = document.lower()
            # Remove punctuation
            document = document.translate(str.maketrans("", "", string.punctuation))
            # Tokenize
            words = word_tokenize(document)
            # Lemmatize
            words = [self.lemmatizer.lemmatize(word) for word in words]
            X_transformed.append(" ".join(words))
        return X_transformed


# create an instance
preprocessor = TextPreprocessor()
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_train)
