from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def model():
    """Returns a vader sentiment analysis model"""
    return SentimentIntensityAnalyzer()