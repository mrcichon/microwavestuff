import numpy as np

def l1_distance(sig1, sig2, normalize=True):
    a, b = sig1.copy(), sig2.copy()
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)
    return np.mean(np.abs(a - b))

def l2_distance(sig1, sig2, normalize=True):
    a, b = sig1.copy(), sig2.copy()
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)
    diff = a - b
    return np.sqrt(np.mean(diff * diff))

def adaptive_distance(sig1, sig2, weights, p):
    diff = np.abs(sig1 - sig2) ** p
    return np.sum(weights * diff) ** (1.0 / p)
