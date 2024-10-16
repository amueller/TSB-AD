import numpy as np
from statsmodels.tsa.seasonal import STL


def series_decompose(X, window_size):
    res = STL(X, period=window_size).fit()
    score = np.abs(res.resid)
    return score / np.max(score)