import numpy as np
from scipy import signal

def compute_shape_matrix(files_data, param, metric='xcorr', normalize=True, max_lag=None, adaptive_params=None):
    """
    Compute shape comparison matrix.
    
    Args:
        files_data: list of dicts with 'name', 'signal', 'freq'
        param: parameter name
        metric: 'xcorr', 'l1', 'l2', 'al1', 'al2'
        normalize: whether to normalize
        max_lag: max lag for cross correlation
        adaptive_params: dict with 'alpha', 'sigma', 'gamma', 'w_min', 'activity_type'
    
    Returns:
        dict with matrix, lag matrix, weights, etc
    """
    n = len(files_data)
    if n < 2:
        return None
    
    # Align signals to same frequency grid
    signals, freqs = _align_signals(files_data)
    
    # Calculate adaptive weights if needed
    weights = None
    activity = None
    if metric in ['al1', 'al2'] and adaptive_params:
        derivs = _compute_derivatives(signals)
        activity = _compute_activity_field(derivs, adaptive_params['alpha'], 
                                          adaptive_params['activity_type'])
        activity = _gaussian_smooth(activity, adaptive_params['sigma'])
        weights = _compute_adaptive_weights(activity, adaptive_params['gamma'], 
                                           adaptive_params['w_min'])
    
    # Compute distance/correlation matrix
    matrix = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                if metric == 'xcorr':
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = 0.0
                lag_matrix[i, j] = 0
            else:
                if metric == 'xcorr':
                    lag, val = compute_cross_correlation_matrix(signals[i], signals[j], 
                                                                normalize, max_lag)
                    matrix[i, j] = matrix[j, i] = val
                    lag_matrix[i, j] = lag_matrix[j, i] = lag
                elif metric == 'l1':
                    val = _l1_distance(signals[i], signals[j], normalize)
                    matrix[i, j] = matrix[j, i] = val
                elif metric == 'l2':
                    val = _l2_distance(signals[i], signals[j], normalize)
                    matrix[i, j] = matrix[j, i] = val
                elif metric == 'al1':
                    val = _adaptive_distance(signals[i], signals[j], weights, 1)
                    matrix[i, j] = matrix[j, i] = val
                elif metric == 'al2':
                    val = _adaptive_distance(signals[i], signals[j], weights, 2)
                    matrix[i, j] = matrix[j, i] = val
    
    return {
        'matrix': matrix,
        'lag_matrix': lag_matrix,
        'weights': weights,
        'activity': activity,
        'frequencies': freqs,
        'names': [f['name'] for f in files_data]
    }


def compute_cross_correlation_matrix(s1, s2, normalize=True, max_lag=None):
    """Compute cross correlation between two signals."""
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


def compute_distance_matrix(files_data, param, metric='l1', normalize=True):
    """Simplified distance computation for L1/L2."""
    return compute_shape_matrix(files_data, param, metric, normalize)


def format_shape_text(shape_data, metric):
    """Format shape comparison results."""
    if shape_data is None:
        return ""
    
    lines = []
    lines.append("Shape Comparison")
    lines.append(f"Metric: {metric}")
    lines.append(f"Files: {len(shape_data['names'])}")
    lines.append("-" * 40)
    
    matrix = shape_data['matrix']
    n = matrix.shape[0]
    vals = matrix[np.triu_indices(n, 1)]
    
    if metric == 'xcorr':
        lines.append(f"Max correlation: {np.max(vals):.3f}")
        lines.append(f"Min correlation: {np.min(vals):.3f}")
        lines.append(f"Mean correlation: {np.mean(vals):.3f}")
    else:
        lines.append(f"Min distance: {np.min(vals):.4f}")
        lines.append(f"Max distance: {np.max(vals):.4f}")
        lines.append(f"Mean distance: {np.mean(vals):.4f}")
    
    return "\n".join(lines)


# Helper functions
def _align_signals(files_data):
    """Align signals to common frequency grid."""
    f0 = files_data[0]['freq']
    signals = []
    
    for data in files_data:
        f = data['freq']
        s = data['signal']
        if len(f) == len(f0) and np.allclose(f, f0, rtol=1e-6):
            signals.append(s)
        else:
            # Interpolate to common grid
            signals.append(np.interp(f0, f, s))
    
    return np.array(signals), f0


def _compute_derivatives(signals):
    """Compute numerical derivatives."""
    n_signals, n_points = signals.shape
    derivs = np.zeros_like(signals)
    
    for i in range(n_signals):
        derivs[i, 1:-1] = (signals[i, 2:] - signals[i, :-2]) / 2
        derivs[i, 0] = signals[i, 1] - signals[i, 0]
        derivs[i, -1] = signals[i, -1] - signals[i, -2]
    
    return derivs


def _compute_activity_field(derivs, alpha, activity_type):
    """Compute activity field from derivatives."""
    if activity_type == "nonincreasing":
        activity_contrib = np.where(derivs <= 0, np.abs(derivs)**alpha, 0)
    else:  # bidirectional
        activity_contrib = np.abs(derivs)**alpha
    
    return np.mean(activity_contrib, axis=0)


def _gaussian_smooth(field, sigma):
    """Apply Gaussian smoothing."""
    if sigma == 0:
        return field
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(field, sigma, mode='reflect')


def _compute_adaptive_weights(activity, gamma, w_min):
    """Compute adaptive weights from activity field."""
    eps = 1e-10
    activity_max = np.max(activity) + eps
    weights = w_min + (1 - w_min) * (activity / activity_max)**gamma
    weights *= len(weights) / np.sum(weights)
    return weights


def _l1_distance(a, b, normalize):
    """L1 distance."""
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)
    return np.mean(np.abs(a - b))


def _l2_distance(a, b, normalize):
    """L2 distance."""
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)
    return np.sqrt(np.mean((a - b)**2))


def _adaptive_distance(s1, s2, weights, p):
    """Adaptive Lp distance."""
    diff = np.abs(s1 - s2)**p
    return np.sum(weights * diff)**(1/p)
