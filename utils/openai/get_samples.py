
def get_samples(df, n_samples=8, label_col="label"):
    """
    Get a random sample of n_samples per class from the dataset.
    The samples are shuffled to avoid recency bias.
    """
    df = df.groupby(label_col, sort=False).apply(lambda x: x.sample(n_samples))
    df = df.reset_index(drop=True)
    df = df.sample(frac=1)
    return df