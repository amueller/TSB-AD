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
    

def plot_detection(signal, label=None, scores=None, train=None, ax=None, linewidth=1, window_length=None, min_anomaly_width=5, percentile=95, score_linestyle="--"):
    if train is not None:
        if signal.index.min() < train.index.max():
            signal.index = signal.index + train.index.max()
    scores = pd.Series(scores, index=signal.index)
    if label is None:
        label = (scores > np.percentile(scores, percentile)).astype(int)
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
        start = label.index[start]
        end = label.index[end]
        width_org = end - start
        width = width_org
        if min_anomaly_width is not None:
            width = np.maximum(width, min_anomaly_width)
        if width > width_org:
            padding = (width - width_org) // 2
            start -= padding
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


def all_peak_analysis(data, n_lags=5000):
    a, b = np.quantile(data, [0.001, 0.999])
    data_clipped = np.clip(data, a, b)
    auto_corr = acf(data_clipped, nlags=n_lags, fft=True)
    peaks, _ = find_peaks(auto_corr)

    prominences = peak_prominences(auto_corr, peaks)[0]
    widths = peak_widths(auto_corr, peaks, rel_height=0.5)[0]
    auto_corr_peaks = auto_corr[peaks]
    width_fraction = widths / peaks
    return {'peaks': peaks,
            'prominences': prominences,
            'widths': widths,
            'auto_corr': auto_corr_peaks,
            'width_fraction': width_fraction,
            'peak_idx': np.arange(len(peaks)),
            'prominence_ranks': stats.rankdata(-prominences),
            'width_fraction_ranks': stats.rankdata(-width_fraction),
            'auto_corr_ranks': stats.rankdata(-auto_corr_peaks)}


def peak_analysis(data, n_lags=5000):
    a, b = np.quantile(data, [0.001, 0.999])
    data_clipped = np.clip(data, a, b)
    auto_corr = acf(data_clipped, nlags=n_lags, fft=True)
    peaks, _ = find_peaks(auto_corr)

    prominences = peak_prominences(auto_corr, peaks)[0]
    widths = peak_widths(auto_corr, peaks, rel_height=0.5)[0]

    most_prominent_idx = np.argmax(prominences)
    most_prominent_peak = peaks[most_prominent_idx]
    most_prominent_prominence = prominences[most_prominent_idx]
    
    first_peak = peaks[0]
    first_prominence = prominences[0]
    first_width = widths[0]
    first_prominence_rank = np.argsort(prominences)[::-1].tolist().index(0)


    return {'prominent_peak': most_prominent_peak,
            'prominent_prominence': most_prominent_prominence, 
            'prominent_width': widths[most_prominent_idx],
            'first_peak': first_peak,
            'first_prominence': first_prominence,
            'first_width': first_width,
            'first_prominence_rank': first_prominence_rank}


def random_walk(length=10000):
    return np.cumsum(np.random.uniform(-1, 1, size=length))

def noise(length=10000):
    return np.random.normal(size=length)

def periodic(length=10000, period=100):
    pattern = random_walk(period)
    ind = np.arange(length)
    return pattern[ind % period]


def make_series(num_periods=1, base_period=True, max_length=10000):
    length = np.random.randint(100, max_length)
    noise_strength = np.random.uniform(0, 10)
    signal = random_walk(length=length) + noise_strength * noise(length=length)
    periods = []
    for period in range(num_periods):
        period_strength = np.random.uniform(0, 10)
        if base_period and len(periods):
            if length //  (10 * periods[0][1]) <= 3:
                break
            period = periods[0][1] * np.random.randint(3, min(2000, length //  (10 * periods[0][1])))
        else:
            period = np.random.randint(3, np.random.randint(5, min(2000, length // 10)))
        periods.append([period_strength, period])
        signal = signal + period_strength * periodic(period=period, length=length)
    return signal, periods, length, noise_strength



def find_length(data, prominence_percentile=90, n_lags=5000, max_filter=False,
                ensure_min_points=False, std_multiplier=2, scale_n_lags=False,
                most_prominent=True):
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
    if most_prominent:
        result = peaks[masked_inds[np.argmax(prominences[masked_inds])]]
    else:
        result = peaks[masked_inds[0]]
        if result < 3:
            result = peaks[masked_inds[1]]
    sorted_prominences = np.sort(prominences)
    if len(prominences) < 2:
        good_max = sorted_prominences[-1]
    else:
        good_max = sorted_prominences[-2]
    prominence_threshold = good_max - std_multiplier * prominences[masked_inds].std()
    if ensure_min_points:
        if len(masked_inds) > 10:
            prominence_threshold = min(prominence_threshold, np.sort(prominences[masked_inds])[-10])
    pruned_inds = masked_inds[prominences[masked_inds] > prominence_threshold]

    if len(pruned_inds) > 2:
        # hard-coded maximum number of peaks to consider as 20
        mode = stats.mode(np.diff(np.sort(peaks[pruned_inds])[:20]))
        # hard-coded minimum periodicity of 5
        if (mode.count > 3 and mode.mode > 3) and (mode.mode in peaks[masked_inds] or mode.mode * 2 in peaks[masked_inds] or mode.mode * 3 in peaks[masked_inds] or mode.mode * 4 in peaks[masked_inds]):
            result = mode.mode
            confirmed = True
        elif mode.count > 1 and mode.mode in peaks[pruned_inds] and mode.mode > 5:
            # usually mode is the first peak but not always.
            confirmed = True
            result = mode.mode
        elif mode.mode > 5:
            diffs = np.diff(np.sort(peaks[pruned_inds])[:20])
            good_diffs = np.abs(diffs - peaks[pruned_inds][0]) / diffs < 0.05  # within 5% of first peak
            if good_diffs.sum() > 2:
                result = int(np.round(diffs[good_diffs].mean()))
                confirmed = True

    max_prominence = np.max(prominences)

    if scale_n_lags:
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

def check_even_fractions(peak_idx, peaks):
    peak = peaks[peak_idx]
    for divisor in [2, 3]:
        if peak % divisor == 0:
            new_peak_idx = np.where(peaks == peak // divisor)[0]
            if len(new_peak_idx):
                peak_idx = new_peak_idx[0]
                peak = peaks[peak_idx]
    return peak_idx


def find_two_lengths(data, n_lags=5000, peak_threshold1=0.01, peak_threshold2=0.1):
    auto_corr = acf(data, nlags=n_lags, fft=True)
    peaks, _ = find_peaks(auto_corr)
    prominences = peak_prominences(auto_corr, peaks)[0]

    if not len(prominences):
        return 0, 0

    masked_inds = np.where(auto_corr[peaks] >= np.maximum.accumulate(auto_corr[peaks][::-1])[::-1])[0]
    result_ind = masked_inds[0]
    result = peaks[result_ind]
    if result < 3:
        result_ind = masked_inds[1]
        result = peaks[result_ind]
    second_res_ind = np.argmax(prominences[masked_inds])
    second_res = peaks[masked_inds[second_res_ind]]
    second_res_prominence = prominences[second_res_ind]

    if second_res == result and result_ind > 1:
        # the first non-dominated peak is also the most prominent peak
        # that likely means any secondary periodicity is dominated by that peak, so smaller than it.
        # so we pick the most prominent peak that's smaller than the found peak
        second_res_ind = np.argmax(prominences[:result_ind])
        second_res_ind = check_even_fractions(second_res_ind, peaks)
        second_res = peaks[second_res_ind]

    if prominences[result_ind] < peak_threshold1:
        result = 0
    if second_res == result or second_res_prominence < peak_threshold2:
        second_res = 0

    return result, second_res