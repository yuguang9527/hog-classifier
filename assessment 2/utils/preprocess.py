# Copyright (C), Weights & Biases
# Please DO NOT share.

import numpy as np


def normalize(data, data_mean=None, data_range=None):
    """Normalizes input data.

    Notice that we have two optional input arguments to this function. When
    dealing with validation/test data, we expect these to be given, since we
    should not learn or change *anything* of the trained model. This includes
    the data preprocessing step.

    Parameters
    ----------
    data : ndarray
        Input data that we want to normalize. NxD, where D is the
        dimension of the input data.

    data_mean : ndarray (optional)
        Mean of the data that we should use. 1xD is expected. If not given,
        this will be computed from `data`.

    data_range : ndarray (optional)
        Maximum deviation from the mean. 1xD is expected. If not given, this
        will be computed from `data`.

    Returns
    -------
    data_n : ndarray
        Normalized data. NxD, where D is the dimension of the input data.

    data_mean : ndarray
        Mean. 1xD.

    data_range : ndarray
        Maximum deviation from the mean. 1xD.

    """

    data_f = data.astype(float)

    # Make zero mean
    if data_mean is None:
        data_mean = np.mean(data_f, axis=0, keepdims=True)
    data_n = data_f - data_mean

    # Make range -1 and 1
    if data_range is None:
        data_range = np.max(np.abs(data_n), axis=0, keepdims=True)
    data_n = data_n / data_range

    return data_n, data_mean, data_range


#
# preprocess.py ends here
