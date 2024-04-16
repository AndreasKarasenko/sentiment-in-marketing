from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

# create an instance of your vectorizer
cv = TfidfVectorizer()  # or use HashingVectorizer
your_documents = ...  # load your documents here

# transform your documents
xformed = cv.fit_transform([your_documents])
...

# create a pipeline instead
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [("tfidf", TfidfVectorizer())]
)  # pipelines can also include a classifier which we show at a later time
# transform your documents
xformed = pipeline.fit_transform([your_documents])
...
