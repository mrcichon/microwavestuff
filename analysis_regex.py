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

    return "\n".join(lines)
