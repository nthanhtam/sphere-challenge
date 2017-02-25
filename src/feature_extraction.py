# -*- coding: utf-8 -*-
import warnings
import numpy as np
from scipy.signal import convolve
from spectrum import *

warnings.filterwarnings('ignore')

# features
def peak(arr):
    """
    Reference: http://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy
    :param arr:
    :return:
    """
    # Obtaining derivative
    kernel = [1, 0, -1]
    dY = convolve(arr, kernel, 'valid')

    # Checking for sign-flipping
    S = np.sign(dY)
    ddS = convolve(S, kernel, 'valid')

    # These candidates are basically all negative slope positions
    # Add one since using 'valid' shrinks the arrays
    candidates = np.where(dY < 0)[0] + (len(kernel) - 1)

    # Here they are filtered on actually being the final such position in a run of
    # negative slopes
    peaks = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))
    return len(peaks)


def zero_crossing(arr):
    """
    refereence: http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    :param arr:
    :return:
    """
    return len(np.where(numpy.diff(np.sign(arr)))[0])


def energy(arr):
    """
    Energy measure. Sum of the squares divided by the number of values.
    :param arr:
    :return: float
    """
    return np.sum(np.power(arr, 2)) / len(arr)


def entropy(arr):
    """
    Reference: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
    :param arr:
    :return:
    """
    lensig = len(arr)
    symset = list(set(arr))
    propab = [np.size(arr[arr == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent


def mad(arr):
    """
    Median Absolute Deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
    http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    :param arr:
    :return: float
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def iqr(arr):
    """
    Interquartile Range: http://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy
    :param arr:
    :return:
    """
    q75, q25 = np.percentile(arr, [75, 25])
    return q75 - q25


def lag(features, forward=False):
    lag_features = np.zeros(features.shape)
    # moving backward/forward
    if forward:
        lag_features[-1, :] = -1
        lag_features[:-1:, :] = features[1:, :]
    else:
        lag_features[0, :] = -1
        lag_features[1:, :] = features[:-1, :]
    return lag_features



