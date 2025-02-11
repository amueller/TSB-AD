from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from scipy.signal import find_peaks, peak_prominences


# determine sliding window (period) based on ACF
def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 0
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    

# # determine sliding window (period) based on ACF, Original version
# def find_length(data):
#     if len(data.shape)>1:
#         return 0
#     data = data[:min(20000, len(data))]
    
#     base = 3
#     auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
#     local_max = argrelextrema(auto_corr, np.greater)[0]
#     try:
#         max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
#         if local_max[max_local_max]<3 or local_max[max_local_max]>300:
#             return 125
#         return local_max[max_local_max]+base
#     except:
#         return 125


def find_length(data, n_lags=5000, max_filter=False,
                ensure_min_points=False, std_multiplier=2,
                most_prominent=True):
    # slightly more hacky but more robust heuristic based on the autocorrelation function
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
     
    return result, confirmed, max_prominence