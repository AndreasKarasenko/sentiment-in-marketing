from lime import lime_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def interpret_model(model, text, class_names: list = ["Negative", "Positive"]):
    # Create a pipeline with TF-IDF vectorizer and the ML model
    pipeline = make_pipeline(TfidfVectorizer(), model)

    # Create an explainer using LIME
    explainer = lime_text.LimeTextExplainer(class_names=class_names)

    # Generate an explanation for the given text
    explanation = explainer.explain_instance(
        text, pipeline.predict_proba, num_features=10
    )

    # Print the top features and their weights
    print("Top features:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight}")
