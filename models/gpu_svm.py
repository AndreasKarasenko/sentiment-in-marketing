from cuml.svm import SVC as cuSVC


def model():
    """Return a support vector machine model."""
    return cuSVC()
