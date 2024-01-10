# function for running the Friedman test on the results of the models

import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi_friedman

def run_friedman_test(*args):
    """
    Run the Friedman test on the results of the models.

    Args:
        *args: Variable length argument list of arrays representing the results of the models.

    Returns:
        The test statistic and the p-value.
    """
    return stats.friedmanchisquare(*args)

def run_posthoc(data):
    """
    Run the post-hoc Nemenyi test on the results of the models.

    data:
        A numpy array representing the results of the models.

    Returns:
        The test statistic and the p-value.
    """
    return posthoc_nemenyi_friedman(data.T)