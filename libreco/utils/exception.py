class NotSamplingError(Exception):
    """Exception related to sampling data

    If client wants to use batch_sampling and then evaluation on the dataset,
    but forgot to do whole data sampling beforehand, this exception will be
    raised. Because in this case, unsampled data can't be evaluated.
    """
    pass

