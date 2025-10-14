import numpy as np
from scipy.ndimage import convolve1d

def cross_correlation(s1, s2, normalize=True, max_lag=None):
    if normalize:
        s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
        s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
    
    n = len(s1) + len(s2) - 1
    S1 = np.fft.fft(s1, n)
    S2 = np.fft.fft(s2, n)
    corr = np.real(np.fft.ifft(S1 * np.conj(S2)))
    corr = np.concatenate([corr[-(len(s2)-1):], corr[:len(s1)]])
    
    lags = np.arange(-len(s2) + 1, len(s1))
    
    if max_lag is not None and max_lag > 0:
        valid_indices = np.abs(lags) <= max_lag
        corr = corr[valid_indices]
        lags = lags[valid_indices]
    
    best_idx = np.argmax(corr)
    
    if normalize:
        corr = corr / (np.sqrt(np.sum(s1**2) * np.sum(s2**2)) + 1e-10)
    
    return lags[best_idx], corr[best_idx]

def align_signals(refs, frs):
    f0 = frs[0]
    out = []
    for s, f in zip(refs, frs):
        if len(f) == len(f0) and np.allclose(f, f0, rtol=1e-6):
            out.append(s)
            continue
        out.append(np.interp(f0, f, s))
    return np.array(out), f0

def compute_derivatives(signals):
    n_signals, n_points = signals.shape
    derivs = np.zeros_like(signals)
    for i in range(n_signals):
        derivs[i, 1:-1] = (signals[i, 2:] - signals[i, :-2]) / 2
        derivs[i, 0] = signals[i, 1] - signals[i, 0]
        derivs[i, -1] = signals[i, -1] - signals[i, -2]
    return derivs

def compute_activity_field(derivs, alpha, activity_type):
    if activity_type == "nonincreasing":
        activity_contrib = np.where(derivs <= 0, np.abs(derivs)**alpha, 0)
    else:
        activity_contrib = np.abs(derivs)**alpha
    return np.mean(activity_contrib, axis=0)

def gaussian_smooth(field, sigma):
    if sigma == 0:
        return field
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(field, sigma, mode='reflect')

def compute_adaptive_weights(activity, gamma, w_min):
    eps = 1e-10
    activity_max = np.max(activity) + eps
    weights = w_min + (1 - w_min) * (activity / activity_max)**gamma
    weights *= len(weights) / np.sum(weights)
    return weights
