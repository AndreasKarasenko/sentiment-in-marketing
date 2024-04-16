## Preprocessing

Before passing our data to a model we first need to put it into a format that the computer can understand. We require numeric data that captures and represents the content of our text. Additionally we are sometimes interested in "cleaning" the data, to make the representations easier and less granular.
This can include, lowercasing, stemming, lemmatization, stop-word removal and more. For the actual numeric representation we can use functions such as TfidfVectorizer, HashVectorizer, or any other text usable vectorizer from sklearn or similar.

### Data Cleaning
An example class for cleaning data is given below.
~~~Python
import string
import zipfile
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
~~~

### Producing numerical representations
The next step we must take is to transform the text into only numbers after which we can use it for modeling.

~~~Python
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

# create an instance of your vectorizer
cv = TfidfVectorizer()  # or use HashingVectorizer
your_documents = ...  # load your documents here

# transform your documents
xformed = cv.fit_transform([your_documents])
...
~~~