from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_model():
    """Returns a vader sentiment analysis model"""
    return SentimentIntensityAnalyzer()