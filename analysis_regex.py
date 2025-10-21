import numpy as np
import re

def extract_regex_value(filename, pattern, group_idx=1):
    """Extract numeric value from filename using regex pattern."""
    try:
        match = re.search(pattern, filename)
        if match and group_idx <= len(match.groups()):
            return float(match.group(group_idx))
    except:
        pass
    return None


def analyze_ordered_ranges(files_data, strict_monotonic=False, threshold=0.1, ref_db=10.0, min_count=1):
    """
    Analyze ordered data for monotonic regions and small difference regions.
    
    Args:
        files_data: list of tuples (filename, value, freq_array, s_data)
        strict_monotonic: if True, require strictly increasing/decreasing
        threshold: threshold for small differences
        ref_db: reference dB level for scaling threshold
        min_count: minimum count of small differences
    
    Returns:
        tuple: (monotonic_ranges, frequencies, s_matrix, labels, small_diff_ranges)
    """
    if not files_data:
        return [], None, None, [], []
    
    files_data.sort(key=lambda x: x[1])
    labels = [d[0] for d in files_data]
    freqs = files_data[0][2]
    s_matrix = np.array([d[3] for d in files_data])
    
    monotonic_indices = []
    for i in range(s_matrix.shape[1]):
        values = s_matrix[:, i]
        diffs = np.diff(values)
        
        is_monotonic = False
        if strict_monotonic:
            if np.all(diffs > 0) or np.all(diffs < 0):
                is_monotonic = True
        else:
            if np.all(diffs >= 0) or np.all(diffs <= 0):
                is_monotonic = True
        
        if is_monotonic:
            monotonic_indices.append(i)
    
    freq_ranges = []
    if monotonic_indices:
        start = monotonic_indices[0]
        for i in range(1, len(monotonic_indices)):
            if monotonic_indices[i] != monotonic_indices[i - 1] + 1:
                end = monotonic_indices[i - 1]
                freq_ranges.append((freqs[start], freqs[end]))
                start = monotonic_indices[i]
        freq_ranges.append((freqs[start], freqs[monotonic_indices[-1]]))
    
    small_diff_ranges = []
    if monotonic_indices:
        for i in monotonic_indices:
            values = s_matrix[:, i]
            diffs = np.diff(values)
            abs_diffs = np.abs(diffs)
            
            avg_level = np.mean(values)
            if ref_db != 0:
                scaled_threshold = threshold * abs(avg_level) / abs(ref_db)
            else:
                scaled_threshold = threshold
            
            scaled_threshold = max(scaled_threshold, threshold * 0.1)
            small_diff_count = np.sum(abs_diffs <= scaled_threshold)
            
            if small_diff_count >= min_count:
                small_diff_ranges.append((freqs[i], freqs[i]))
    
    merged_small_diff_ranges = []
    if small_diff_ranges:
        start_freq = small_diff_ranges[0][0]
        end_freq = small_diff_ranges[0][1]
        
        for i in range(1, len(small_diff_ranges)):
            curr_start, curr_end = small_diff_ranges[i]
            freq_idx_prev = np.where(freqs == end_freq)[0][0]
            freq_idx_curr = np.where(freqs == curr_start)[0][0]
            
            if freq_idx_curr == freq_idx_prev + 1:
                end_freq = curr_end
            else:
                merged_small_diff_ranges.append((start_freq, end_freq))
                start_freq = curr_start
                end_freq = curr_end
        
        merged_small_diff_ranges.append((start_freq, end_freq))
    
    return freq_ranges, freqs, s_matrix, labels, merged_small_diff_ranges


def find_monotonic_regions(values, strict=False):
    """Find indices where values are monotonic."""
    diffs = np.diff(values)
    if strict:
        return np.all(diffs > 0) or np.all(diffs < 0)
    else:
        return np.all(diffs >= 0) or np.all(diffs <= 0)


def find_small_diff_regions(values, threshold, ref_value):
    """Find regions with small differences."""
    diffs = np.abs(np.diff(values))
    if ref_value != 0:
        scaled_threshold = threshold * abs(np.mean(values)) / abs(ref_value)
    else:
        scaled_threshold = threshold
    return diffs <= scaled_threshold


def format_regex_text(pattern, group_idx, freq_range, ranges, files_info):
    """Format regex analysis results as text."""
    lines = []
    lines.append(f"Regex: {pattern} (group {group_idx})")
    lines.append(f"Frequency range: {freq_range[0]}-{freq_range[1]} GHz")
    lines.append("-" * 40)
    
    for info in files_info:
        lines.append(info)
    
    if ranges:
        lines.append("")
        lines.append("Monotonic ranges found:")
        for start, end in ranges:
            lines.append(f"  {start/1e9:.3f} - {end/1e9:.3f} GHz")
    
    return "\n".join(lines)
