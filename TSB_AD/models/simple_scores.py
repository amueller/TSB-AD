from functools import partial
import pandas as pd
import numpy as np
from joblib import Memory

from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest
import warnings

from numba.core.errors import NumbaPerformanceWarning
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks, peak_prominences


memory = Memory(location=".")


def find_length(data, n_lags=5000, strict=False):
    default_length = 125
    # very simple heuristic based on the autocorrelation function
    auto_corr = acf(data, nlags=n_lags, fft=True)
    peaks, _ = find_peaks(auto_corr)

    if not len(peaks):
        return default_length

    masked_inds = np.where(auto_corr[peaks] >= np.maximum.accumulate(auto_corr[peaks][::-1])[::-1])[0]
    result_idx = masked_inds[0]
    result = peaks[result_idx]
    if result < 3:
        result_idx = masked_inds[1]
        result = peaks[result_idx]
    if strict:
        prominences = peak_prominences(auto_corr, peaks)[0]
        if prominences[result_idx] < 0.01:
            return default_length

    if 4 * result > len(data):
        result = result // 8
    return result


def signal_score(series, train_ignored=None):
    length = find_length(series)
    return series, length


def random_score(series, train_ignored=None):
    length = find_length(series)
    return np.random.uniform(size=len(series)), length


def neg_signal_score(series, train_ignored=None):
    length = find_length(series)
    return -series, length


def rolling_score(series, train_ignored=None, window=None, center=True):
    length = find_length(series)
    window = window or length
    return series.rolling(window, center=center).mean().fillna(series.mean()), length


def diff_rolling_score(series, train_ignored=None, window=None, center=True):
    length = find_length(series)
    window = window or length
    return (series - series.rolling(window, center=center).mean().fillna(series.mean())) ** 2, length


def mean_dist_score(series, train_ignored=None):
    length = find_length(series)
    return (series - series.mean()).abs(), length


def median_dist_score(series, train_ignored=None):
    length = find_length(series)
    return (series - series.median()).abs(), length


def median_dist_rolling_score(series, train_ignored=None, center=True, window=None):
    length = find_length(series)
    window = window or length
    return (series.rolling(window, center=center).mean().fillna(series.mean()) - series.median()).abs(), length


def median_dist_rolling_score2(series, train_ignored=None, center=True, window=None):
    length = find_length(series)
    window = window or length
    rolling = series.rolling(window, center=center).mean().fillna(series.mean())
    return (rolling - rolling.median()).abs(), length


def neg_rolling_score(series, train_ignored=None, rolling=None, center=True):
    length = find_length(series)
    rolling = rolling or length
    return -series.rolling(rolling, center=center).mean().fillna(series.mean()), length


def rolling_diff_score(series, train_ignored=None, window1=1, window2=3, center=True, multiplicative=False):
    length = find_length(series)
    if multiplicative:
        window1 = int(window1 * length)
        window2 = int(window2 * length)
    rolling1 = series.rolling(window1, center=center).mean().fillna(series.mean())
    rolling2 = series.rolling(window2, center=center).mean().fillna(series.mean())
    return (rolling1 - rolling2) ** 2, length


def diff_diff_std_score(series, train_ignored=None):
    length = find_length(series)
    scores = series.diff().diff().rolling(length, center=True).std().ffill().bfill()
    scores = (scores - scores.mean()).abs()
    return scores, length


def average_score(series, train_ignored=None, score_funcs=None, normalize="sum"):
    scores = pd.Series(0, series.index)
    for score_func in score_funcs:
        scores_new, length = score_func(series)
        if normalize == "sum":
            scores_new = scores_new / scores_new.sum()
        elif normalize == "scale":
            scores_new = (scores_new - scores_new.mean())/scores_new.std()
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize value: {normalize}")
        scores += scores_new
    return scores, length


def robust_diff_std(series, train_ignored=None, window1=None):
    length = find_length(series)
    window1 = window1 or length
    scores = series.diff().diff().abs().rolling(window1, center=True).std().fillna(0)
    scores = (scores / scores.rolling(length, center=True).mean()).ffill().bfill()
    return scores, length



@memory.cache
def discord_score(test, train=None, normalize=True):
    import stumpy
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
    if train is not None:
        length = find_length(train)
        mp = stumpy.gpu_stump(test, length, train, ignore_trivial=False, normalize=normalize)
    else:
        length = find_length(test)
        mp = stumpy.gpu_stump(test, length, ignore_trivial=True, normalize=normalize)
    res = np.zeros(len(test))
    res.fill(mp[:, 0].min())
    res[length//2:-length//2+1] = mp[:, 0]
    return res, length


kpi_averaged_score = partial(average_score, score_funcs=[
        median_dist_score,
        partial(rolling_diff_score, window1=3, window2=150),
        partial(robust_diff_std, window1=102),
        ])

ucr_combo = partial(average_score, score_funcs=[
        diff_diff_std_score,
        partial(rolling_diff_score, window1=1, window2=5, multiplicative=True),
        ])


simple_scores = [(rolling_score, "rolling window score"),
                 (median_dist_rolling_score, "median dist rolling score"),
                 (diff_rolling_score, "diff rolling score"),
                 (signal_score, "raw signal"),
                 (neg_signal_score, "negative raw signal"),
                 (neg_rolling_score, "negative rolling window score"),
                 (mean_dist_score, "mean dist score"),
                 (median_dist_score, "median dist score"),
                 (diff_diff_std_score, "diff diff std score"),
                 (partial(rolling_score, window=3), "rolling 3"),
                 # (robust_diff_std, "robust diff std"),
                 # (partial(robust_diff_std, window1=100), "robust diff std 100"),
                 # (partial(rolling_diff_score, window1=8, window2=100), "rolling diff 8, 100"),
                 # (partial(rolling_diff_score, window1=3, window2=150), "rolling diff 3, 150"),
                 # (partial(rolling_diff_score, window1=0.1, window2=0.01, multiplicative=True), "rolling diff mult 0.1 0.01"),
                 (partial(rolling_diff_score, window1=1, window2=5, multiplicative=True), "rolling diff mult 1 5"),
                 (kpi_averaged_score, "kpi custom average"),
                 (ucr_combo, "ucr custom average"),
                ]