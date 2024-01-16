### functions for descriptive analysis
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def describe(df: pd.DataFrame, text_col: str = "text", score_col: str = "score", plot: bool = False):
    """Descriptive analysis for a pandas DataFrame.

    Args:
        df: pandas DataFrame with 'text' and 'score' columns
        plot: whether to plot the results
    """
    # get number of rows
    n_rows = df.shape[0]

    # get number of missing values in 'text' and 'score' columns
    n_missing_text = df[text_col].isnull().sum()
    n_missing_score = df[score_col].isnull().sum()

    # get number of unique values in 'text' and 'score' columns
    n_unique_text = df[text_col].nunique()
    n_unique_score = df[score_col].nunique()

    # get data types of 'text' and 'score' columns
    dtype_text = df[text_col].dtype
    dtype_score = df[score_col].dtype

    # get descriptive statistics for 'score' column
    stats_score = df[score_col].describe()
    
    # get percentage distribution of 'score' column
    score_distribution = df[score_col].value_counts(normalize=True)
    score_distribution_percentage = score_distribution * 100

    # calculate skewness and kurtosis for 'score' column
    skew_score = df[score_col].skew()
    kurt_score = df[score_col].kurt()

    # determine if 'score' column is skewed, kurtotic, an outlier, has high cardinality, or is highly correlated
    skewed = skew_score > 1
    kurtotic = kurt_score > 1
    outlier = skewed & kurtotic
    high_cardinality = n_unique_score > n_rows / 2
    highly_correlated = stats_score["std"] > stats_score["mean"]

    # if plot is True, plot a histogram of the 'score' column
    if plot:
        plt.hist(df[score_col].dropna())
        plt.title("Histogram of Score")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

    return {
        "n_rows": n_rows,
        "n_missing_text": n_missing_text,
        "n_missing_score": n_missing_score,
        "n_unique_text": n_unique_text,
        "n_unique_score": n_unique_score,
        "dtype_text": dtype_text,
        "dtype_score": dtype_score,
        "stats_score": stats_score,
        "distribution_score": score_distribution_percentage,
        "skew_score": skew_score,
        "kurt_score": kurt_score,
        "skewed": skewed,
        "kurtotic": kurtotic,
        "outlier": outlier,
        "high_cardinality": high_cardinality,
        "highly_correlated": highly_correlated,
    }