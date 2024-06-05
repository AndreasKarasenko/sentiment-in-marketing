
def get_samples(df, n_samples=8, label_col="label"):
    """
    Get a random sample of n_samples per class from the dataset.
    """
    df = df.groupby(label_col).apply(lambda x: x.sample(n_samples))
    df = df.reset_index(drop=True)
    return df