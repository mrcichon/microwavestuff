import numpy as np
import re
from itertools import combinations


def extract_regex_value(filename, pattern, group_idx=1):
    try:
        match = re.search(pattern, filename)
        if match and group_idx <= len(match.groups()):
            return float(match.group(group_idx))
    except:
        pass
    return None


def _mono_mask(sub, strict):
    diffs = np.diff(sub, axis=0)
    if strict:
        return np.all(diffs > 0, axis=0) | np.all(diffs < 0, axis=0)
    return np.all(diffs >= 0, axis=0) | np.all(diffs <= 0, axis=0)


def find_best_drop(s_matrix, n_drop, strict):
    n_curves = s_matrix.shape[0]
    if n_drop == 0:
        return _mono_mask(s_matrix, strict), []
    if n_drop >= n_curves - 1:
        return np.ones(s_matrix.shape[1], dtype=bool), list(range(n_curves))
    best_count = -1
    best_mask = None
    best_drop = []
    for drop in combinations(range(n_curves), n_drop):
        keep = [i for i in range(n_curves) if i not in drop]
        mask = _mono_mask(s_matrix[keep], strict)
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_mask = mask
            best_drop = list(drop)
    return best_mask, best_drop


def compute_kendall_tau(s_matrix):
    n_curves = s_matrix.shape[0]
    concordant = np.zeros(s_matrix.shape[1])
    discordant = np.zeros(s_matrix.shape[1])
    for i in range(n_curves):
        for j in range(i + 1, n_curves):
            diff = s_matrix[j] - s_matrix[i]
            concordant += (diff > 0)
            discordant += (diff < 0)
    total = concordant + discordant
    return np.where(total > 0, (concordant - discordant) / total, 0.0)


def compute_max_displacement(s_matrix):
    n_curves = s_matrix.shape[0]
    expected = np.arange(n_curves)
    ranks = np.argsort(np.argsort(s_matrix, axis=0), axis=0)
    return np.max(np.abs(expected[:, None] - ranks), axis=0)


def mask_to_ranges(mask, freqs):
    if not mask.any():
        return []
    indices = np.where(mask)[0]
    ranges = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            ranges.append((freqs[start], freqs[indices[i - 1]]))
            start = indices[i]
    ranges.append((freqs[start], freqs[indices[-1]]))
    return ranges


def compute_small_diffs(s_matrix, freqs, mono_mask, threshold, ref_db, min_count):
    indices = np.where(mono_mask)[0]
    small_mask = np.zeros(len(freqs), dtype=bool)
    for i in indices:
        values = s_matrix[:, i]
        diffs = np.abs(np.diff(values))
        avg_level = np.mean(values)
        scaled_threshold = threshold * abs(avg_level) / abs(ref_db) if ref_db != 0 else threshold
        scaled_threshold = max(scaled_threshold, threshold * 0.1)
        if np.sum(diffs <= scaled_threshold) >= min_count:
            small_mask[i] = True
    return small_mask


def mask_to_index_ranges(mask):
    if not mask.any():
        return []
    idx = np.where(mask)[0]
    ranges = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            ranges.append((start, idx[i - 1]))
            start = idx[i]
    ranges.append((start, idx[-1]))
    return ranges


def _best_lag(ref, sig, max_lag):
    # integer shift delta such that sig[j+delta] ~ ref[j], minimizing L2 over the
    # overlap after removing each side's mean (so vertical offset doesn't matter)
    n = len(ref)
    max_lag = min(max_lag, n - 2)
    if max_lag < 1:
        return 0
    best_lag, best_err = 0, np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r, s = ref[:n - lag], sig[lag:]
        else:
            r, s = ref[-lag:], sig[:n + lag]
        if len(r) < 2:
            continue
        err = np.mean(((r - r.mean()) - (s - s.mean())) ** 2)
        if err < best_err:
            best_err, best_lag = err, lag
    return best_lag


def _window_lags(W, max_lag, ref_idx=None):
    n = W.shape[0]
    ref = W[n // 2 if ref_idx is None else ref_idx]
    return np.array([_best_lag(ref, W[i], max_lag) for i in range(n)])


def _align_with_lags(W, lags):
    # shift each curve by its lag and return the common core. trim is the actual
    # max |lag| used (not the search bound), so the window only needs to exceed the
    # real shift span, not 2*max_lag.
    n, w = W.shape
    trim = int(np.max(np.abs(lags))) if len(lags) else 0
    core = w - 2 * trim
    if core < 2:
        return W
    out = np.empty((n, core))
    for i in range(n):
        s = trim + lags[i]
        out[i] = W[i, s: s + core]
    return out


def _align_window(W, max_lag):
    # align every curve in the window to the middle curve by integer frequency shift,
    # return the common core
    if W.shape[1] < 4:
        return W
    return _align_with_lags(W, _window_lags(W, max_lag))


def _monotonic_frac(lags, values):
    order = np.argsort(values)
    d = np.diff(np.asarray(lags)[order])
    if len(d) == 0:
        return 1.0
    return max(np.sum(d >= 0), np.sum(d <= 0)) / len(d)


def compute_shift_tracking(s_matrix, values, freqs, window=41, max_lag=60,
                           shape_thresh=0.9, mono_thresh=0.9, min_shift_samples=3):
    # find frequency regions where the curves are ONE feature translating monotonically
    # with the regex value (same shape, different position), then recover the shift law.
    n, m = s_matrix.shape
    df = float(freqs[1] - freqs[0]) if m > 1 else 0.0
    mask = np.zeros(m, dtype=bool)
    h = max(1, window // 2)
    ref_idx = n // 2
    for k in range(m):
        lo, hi = max(0, k - h), min(m, k + h + 1)
        W = s_matrix[:, lo:hi]
        if W.shape[1] <= 2 * max_lag + 2:
            continue
        lags = _window_lags(W, max_lag, ref_idx)
        if lags.max() - lags.min() < min_shift_samples:   # must actually diverge in position
            continue
        if _monotonic_frac(lags, values) < mono_thresh:   # shift ordered by the parameter
            continue
        Wal = _align_with_lags(W, lags)
        if Wal.shape[1] < window // 3:                    # not enough overlap left to judge shape
            continue
        if _congruence_score(Wal) < shape_thresh:         # shape retained after the shift
            continue
        mask[k] = True

    region = None
    iranges = mask_to_index_ranges(mask)
    if iranges:
        a, b = max(iranges, key=lambda r: r[1] - r[0])     # the dominant sliding region
        seg = s_matrix[:, a:b + 1]
        ref = seg[ref_idx]
        shifts_hz = np.array([_best_lag(ref, seg[i], max_lag) for i in range(n)]) * df
        v = np.asarray(values, float)
        if np.ptp(v) > 0:
            slope, intercept = np.polyfit(v, shifts_hz, 1)
            pred = slope * v + intercept
            ss_tot = np.sum((shifts_hz - shifts_hz.mean()) ** 2)
            r2 = 1.0 - np.sum((shifts_hz - pred) ** 2) / ss_tot if ss_tot > 0 else 1.0
        else:
            slope, intercept, r2 = 0.0, 0.0, 0.0
        region = {'f0': float(freqs[a]), 'f1': float(freqs[b]),
                  'shifts_hz': shifts_hz, 'values': v,
                  'slope_hz_per_unit': float(slope), 'intercept': float(intercept), 'r2': float(r2)}
    return mask, region


def _congruence_score(W):
    # fraction of de-leveled shape variation that is shared across all curves.
    # ==1 when curves are exact translates of one shape, ->0 when shapes differ.
    row = W.mean(axis=1, keepdims=True)
    C = W - row
    denom = np.sum(C * C)
    if denom < 1e-12:
        return 1.0
    R = C - C.mean(axis=0, keepdims=True)
    return 1.0 - np.sum(R * R) / denom


def compute_congruence(s_matrix, window=21, threshold=0.9, mode="vertical", max_lag=10):
    n, m = s_matrix.shape
    score = np.ones(m)
    h = max(1, window // 2)
    for k in range(m):
        lo, hi = max(0, k - h), min(m, k + h + 1)
        W = s_matrix[:, lo:hi]
        if mode == "horizontal":
            W = _align_window(W, max_lag)
        score[k] = _congruence_score(W)
    return score, score >= threshold


def congruence_template(segment, mode="vertical", max_lag=10):
    # shared shape mu + g_k over a congruent range; returns (values, left_offset)
    # where left_offset is how many columns were trimmed off the front (horizontal mode)
    W = _align_window(segment, max_lag) if mode == "horizontal" else segment
    mu = W.mean()
    g = W.mean(axis=0) - mu
    off = (segment.shape[1] - W.shape[1]) // 2
    return mu + g, off


def format_regex_text(pattern, group_idx, freq_range, results):
    lines = []
    lines.append(f"Regex: {pattern} (group {group_idx})")
    lines.append(f"Frequency range: {freq_range[0]}-{freq_range[1]} GHz")
    lines.append("-" * 40)

    for info in results.get('files_info', []):
        lines.append(info)

    dropped = results.get('dropped_labels', [])
    if dropped:
        lines.append("")
        lines.append(f"Dropped curves ({len(dropped)}): {', '.join(dropped)}")

    n_pts = results.get('n_freq_points', 0)

    mono = results.get('mono_ranges', [])
    mono_cov = results.get('mono_coverage', 0)
    lines.append("")
    lines.append(f"Monotonic ranges ({len(mono)}, {mono_cov:.1f}% coverage):")
    for s, e in mono:
        lines.append(f"  {s/1e9:.3f} - {e/1e9:.3f} GHz")

    if 'tau_ranges' in results:
        tau_ranges = results['tau_ranges']
        tau_cov = results.get('tau_coverage', 0)
        thresh = results.get('tau_threshold', 0.8)
        lines.append("")
        lines.append(f"Kendall |tau| >= {thresh} ({len(tau_ranges)} ranges, {tau_cov:.1f}% coverage):")
        for s, e in tau_ranges:
            lines.append(f"  {s/1e9:.3f} - {e/1e9:.3f} GHz")

    if 'disp_ranges' in results:
        disp_ranges = results['disp_ranges']
        disp_cov = results.get('disp_coverage', 0)
        thresh = results.get('disp_threshold', 1)
        lines.append("")
        lines.append(f"Max displacement <= {thresh} ({len(disp_ranges)} ranges, {disp_cov:.1f}% coverage):")
        for s, e in disp_ranges:
            lines.append(f"  {s/1e9:.3f} - {e/1e9:.3f} GHz")

    if 'congru_ranges' in results:
        cr = results['congru_ranges']
        cov = results.get('congru_coverage', 0)
        thresh = results.get('congru_threshold', 0.9)
        mode = results.get('congru_mode', 'vertical')
        mode_txt = "vertical offset" if mode == "vertical" else "frequency shift"
        lines.append("")
        lines.append(f"Shared shape ({mode_txt}), congruence >= {thresh} "
                     f"({len(cr)} ranges, {cov:.1f}% coverage):")
        for s, e in cr:
            lines.append(f"  {s/1e9:.3f} - {e/1e9:.3f} GHz")

    tr = results.get('track')
    if tr:
        lines.append("")
        lines.append(f"Sliding feature: {tr['f0']/1e9:.3f} - {tr['f1']/1e9:.3f} GHz")
        lines.append(f"  shift law: {tr['slope_hz_per_unit']/1e6:.2f} MHz / unit "
                     f"(R^2 = {tr['r2']:.3f})")
        for v, s in zip(tr['values'], tr['shifts_hz']):
            lines.append(f"    {v:g} -> {s/1e6:+.1f} MHz")

    return "\n".join(lines)
