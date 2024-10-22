import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy import stats


def load_series(files):
    results = {}
    for f in files:
        if not f.endswith(".csv"):
            f = f + ".csv"
        df = pd.read_csv("benchmark_exp/TSB-AD/TSB-AD-U/" + f)
        results[f.split(".")[0]] = df
    return results
    

def plot_detection(signal, label, scores=None, train=None, ax=None, linewidth=1, window_length=None):
    if train is not None:
        if signal.index.min() < train.index.max():
            signal.index = signal.index + train.index.max()
    scores = pd.Series(scores, index=signal.index)
    label = pd.Series(np.array(label), index=signal.index)
    if ax is None:
        plt.figure(figsize=(40, 5), dpi=300)
        signal_ax = plt.gca()
    else:
        signal_ax = ax
    signal_ax.set_ylabel("signal")
    a, = signal_ax.plot(signal, label='signal', c='k', linewidth=linewidth)
    b = None
    if train is not None:
        signal_ax.plot(train, label='train', c='grey', linewidth=linewidth)
    if scores is not None:
        scores_ax = plt.twinx(signal_ax)
        b, = scores_ax.plot(scores, label='scores', c='b', alpha=0.5, linewidth=linewidth)
        scores_ax.set_ylabel("scores")
    ylims = signal_ax.get_ylim()
    yrange = ylims[1] - ylims[0]
    ymin = ylims[0] - 0.1 * yrange
    ymax = ylims[1] + 0.1 * yrange
    signal_ax.set_ylim(ymin, ymax)
    for start, end in get_anomaly_regions(label):
        width = end - start
        thin_thresh = len(label) / 1e3
        width = np.maximum(width, thin_thresh)
        signal_ax.add_patch(patches.Rectangle((start, ylims[0]), width, ylims[1] - ylims[0], facecolor='red', alpha=0.4))
    red_patch = patches.Patch(color='red', label='anomaly', alpha=0.3)
    plt.legend(handles=[a, b, red_patch] if b is not None else [a, red_patch])
    if window_length is not None:
        locator = MultipleLocator(window_length)
        locator.MAXTICKS = 2000
        signal_ax.xaxis.set_minor_locator(locator)
    return signal_ax


def get_anomaly_regions(labels):
    anomaly_starts = np.where(np.diff(labels) == 1)[0] + 1
    anomaly_ends, = np.where(np.diff(labels) == -1)
    if len(anomaly_ends):
        if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
            # we started with an anomaly, so the start of the first anomaly is the start of the lables
            anomaly_starts = np.concatenate([[0], anomaly_starts])
    if len(anomaly_starts):
        if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
            # we ended on an anomaly, so the end of the last anomaly is the end of the labels
            anomaly_ends = np.concatenate([anomaly_ends, [len(labels) - 1]])
    return list(zip(anomaly_starts, anomaly_ends))


def find_length(data, prominence_percentile=90, n_lags=5000, max_filter=False):
    a, b = np.quantile(data, [0.001, 0.999])
    data_clipped = np.clip(data, a, b)
    auto_corr = acf(data_clipped, nlags=n_lags, fft=True)
    peaks, _ = find_peaks(auto_corr)

    prominences = peak_prominences(auto_corr, peaks)[0]
    confirmed = False
    if not len(prominences):
        return 0, confirmed, 0

    # easy mode assumption, mostly if there's only one periodicity
    if max_filter:
        masked_inds = np.where(auto_corr[peaks] >= np.maximum.accumulate(auto_corr[peaks][::-1])[::-1])[0]
    else:
        masked_inds = np.arange(len(peaks))
    result = peaks[masked_inds[np.argmax(prominences[masked_inds])]]
    sorted_prominences = np.sort(prominences)
    if len(prominences) < 2:
        good_max = sorted_prominences[-1]
    else:
        good_max = sorted_prominences[-2]
    prominence_threshold = good_max - 2 * prominences[masked_inds].std()
    pruned_inds = masked_inds[prominences[masked_inds] > prominence_threshold]

    if len(pruned_inds) > 2:
        # hard-coded maximum number of peaks to consider as 20
        mode = stats.mode(np.diff(np.sort(peaks[pruned_inds])[:20]))
        # hard-coded minimum periodicity of 5
        if (mode.count > 3 and mode.mode > 5):
            result = mode.mode
            confirmed = True
        elif mode.count > 1 and mode.mode in peaks[pruned_inds]:
            # usually mode is the first peak but not always.
            confirmed = True
            result = mode.mode
        else:
            diffs = np.diff(np.sort(peaks[pruned_inds])[:20])
            good_diffs = np.abs(diffs - peaks[pruned_inds][0]) / diffs < 0.05  # within 5% of first peak
            if good_diffs.sum() > 2:
                result = int(np.round(diffs[good_diffs].mean()))
                confirmed = True

    max_prominence = np.max(prominences)

    if 4 * result > n_lags and 4 * result < len(data):
        # we didn't see enough lags for robust detection
        result, confirmed, max_prominence = find_length(data, prominence_percentile=prominence_percentile,
                                        n_lags=result * 4, max_filter=max_filter)         
    
    return result, confirmed, max_prominence


def find_length_diff(data, prominence_percentile=90, n_lags=5000):
    a, b = np.quantile(data, [0.001, 0.999])
    data_clipped = np.clip(data, a, b)
    auto_corr = acf(np.diff(data_clipped), nlags=n_lags, fft=True)
    auto_corr[:2] = 0

    thresh = np.sort(auto_corr)[-10]
    return np.where(auto_corr >= thresh)[0][0] 


    # sorted_peaks = np.argsort(prominences)[::-1]
    # pruned_peaks = sorted_peaks[:max_peaks]
    # widths_threshold = np.percentile(widths, width_percentile)
    # pruned_peaks = pruned_peaks[prominences[pruned_peaks] > prominence_threshold]
    # pruned_peaks = pruned_peaks[widths[pruned_peaks] > widths_threshold]

    # mode = stats.mode(np.diff(np.sort(peaks[pruned_peaks])))
    # if mode.count > 1 and mode.mode in peaks[sorted_peaks[:max_peaks]] or mode.count > 3:
    #     result = mode.mode
    # else:
    #     result = peaks[prominent_peak_idx]

    # if prominences[prominent_peak_idx] < prominence_threshold:
    #     result = [0]
    # else:
    #     highest_peak = np.argmax(auto_corr[peaks])
    #     ac_of_prominent = auto_corr[peaks[prominent_peak_idx]]
    #     # 99% significance level of autocorrelation
    #     if ac_of_prominent < 2.576 / np.sqrt(len(data)):
    #         result, prominences_returned = [0], [0]
    #     else:
    #         result = [peaks[prominent_peak_idx]]
    #         prominences_returned = [prominences[prominent_peak_idx]]
    #         if highest_peak != prominent_peak_idx and prominences[highest_peak] > prominence_threshold:
    #             result.append(peaks[highest_peak])
    #             prominences_returned.append(prominences[highest_peak])
    # return result