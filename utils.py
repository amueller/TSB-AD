import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


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
        locator.MAXTICKS = 5000
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


def find_length(data):
    a, b = np.quantile(data, [0.001, 0.999])
    data = np.clip(data, a, b)
    
    #data = data[:min(20000, len(data))]
    n_lags = 5000
    base = 3
    auto_corr = acf(data, nlags=n_lags, fft=True)[base:]
    try:
        peaks, _ = find_peaks(auto_corr)
        prominences = peak_prominences(auto_corr, peaks)[0]
        peaks = peaks + base
        prominent_peak_idx = np.argmax(prominences)
    except:
        return []
    if prominences[prominent_peak_idx] < 0.1:
        result = []
    else:
        highest_peak = np.argmax(auto_corr[peaks - base])
        ac_of_prominent = auto_corr[peaks[prominent_peak_idx] - base]
        # 99% significance level of autocorrelation
        if ac_of_prominent < 2.576 / np.sqrt(len(data)):
            return []
        result = [peaks[prominent_peak_idx]]
        if highest_peak != prominent_peak_idx and prominences[highest_peak] > 0.1:
            result.append(peaks[highest_peak])
    return result