# Copyright (C), Weights & Biases
# Please DO NOT share.

import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import hog


def extract_h_histogram(data):
    """Extract Hue Histograms from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    h_hist : ndarray (float)
        Hue histgram per image, extracted and reshaped to be NxD, where D is
        the number of bins of each histogram.

    """

    # Convert data into hsv's, and take only the h
    hue = np.asarray([
        rgb2hsv(_x)[:, :, 0] for _x in data
    ])

    # Create bins to be used
    bins = np.linspace(0, 1, 17)

    # Create histogram
    h_hist = []
    for _h in hue:
        hist, _ = np.histogram(_h, bins)
        h_hist += [hist]
    h_hist = np.array(h_hist)

    return h_hist


def extract_hog(data):
    """Extract HOG from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    hog_feat : ndarray (float)
        HOG features per image, extracted and reshaped to be NxD, where D is
        the dimension of HOG features.

    """

    # Using HOG
    # HOG -- without the visualization flag
    hog_feat = np.asarray([
        hog(_x.mean(axis=-1), block_norm="L1") for _x in data 
    ])
    # Check that the above process is not wrong
    # plt.figure()
    # _hog_tmp, _hog_img = hog(data[0].mean(axis=-1), visualise=True)
    # plt.imshow(_hog_img, cmap=plt.cm.gray)
    # plt.show()

    return hog_feat.astype(float).reshape(len(data), -1)


#
# features.py ends here
