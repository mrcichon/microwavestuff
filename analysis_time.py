import numpy as np
from pathlib import Path

def extract_time_data(files_list, freq_range_str, selected_param, use_db=True):
    """
    Extract time domain data for single S-parameter.
    
    Args:
        files_list: list of (BooleanVar, path, metadata_dict)
        freq_range_str: e.g. "0.4-4.0ghz"
        selected_param: str like 's11' or 's21'
        use_db: bool, if True use dB scale, else linear magnitude
    
    Returns:
        list of dicts with keys:
            'name': str
            'path': str
            'freq': ndarray (Hz)
            'freq_data': ndarray (dB/mag values in freq domain)
            't_ns': ndarray (time in ns)
            'time_data': ndarray (dB/mag values in time domain)
            'network': rf.Network (for gating)
            'color': str or None
            'linewidth': float
    """
    from sparams_io import loadFile
    
    result = []
    
    for v, p, d in files_list:
        if not v.get():
            continue
        
        ext = Path(p).suffix.lower()
        
        if not (d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']):
            continue
        
        try:
            ntw_full = d.get('ntwk_full')
            if ntw_full is None:
                ntw_full = loadFile(p)
                d['ntwk_full'] = ntw_full
            
            cached_range = d.get('cached_range')
            if cached_range != freq_range_str:
                ntw = ntw_full[freq_range_str]
                d['ntwk'] = ntw
                d['cached_range'] = freq_range_str
            else:
                ntw = d['ntwk']
            
            fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
            
            s_param = getattr(ntw, selected_param)
            
            freq_data = s_param.s_db.flatten() if use_db else s_param.s_mag.flatten()
            t_ns = s_param.frequency.t_ns
            time_data = s_param.s_time_db.flatten() if use_db else s_param.s_time_mag.flatten()
            
            result.append({
                'name': fname,
                'path': p,
                'freq': ntw.f,
                'freq_data': freq_data,
                't_ns': t_ns,
                'time_data': time_data,
                'network': ntw,
                'color': d.get('line_color'),
                'linewidth': d.get('line_width', 1.0)
            })
            
        except Exception:
            pass
    
    return result


def apply_time_gate(files_data, selected_param, center_ns, span_ns, use_db=True):
    """
    Apply time gating to frequency domain data.
    
    Args:
        files_data: list from extract_time_data()
        selected_param: str like 's11'
        center_ns: float, gate center in ns
        span_ns: float, gate span in ns
        use_db: bool, if True use dB scale, else linear magnitude
    
    Returns:
        list of dicts with keys:
            'name': str
            'freq': ndarray
            'gated_data': ndarray (dB/mag values after gating)
            'color': str or None
            'linewidth': float
    """
    result = []
    
    for file_entry in files_data:
        try:
            ntw = file_entry['network']
            s_param = getattr(ntw, selected_param)
            
            s_gated = s_param.time_gate(center=center_ns, span=span_ns)
            gated_data = s_gated.s_db.flatten() if use_db else s_gated.s_mag.flatten()
            
            result.append({
                'name': file_entry['name'],
                'freq': file_entry['freq'],
                'gated_data': gated_data,
                'color': file_entry['color'],
                'linewidth': file_entry['linewidth']
            })
            
        except Exception:
            pass
    
    return result


def find_time_extrema(files_data, selected_param, freq_range_ghz, 
                      find_minima=True, find_maxima=True):
    """
    Find extrema in frequency domain (raw data).
    
    Args:
        files_data: list from extract_time_data()
        selected_param: str
        freq_range_ghz: tuple (min_ghz, max_ghz)
        find_minima: bool
        find_maxima: bool
    
    Returns:
        list of dicts with keys:
            'type': 'min' or 'max'
            'freq': frequency in Hz
            'value': value in dB
            'param': parameter name
            'file': file name
            'domain': 'freq_raw'
    """
    range_min_hz = freq_range_ghz[0] * 1e9
    range_max_hz = freq_range_ghz[1] * 1e9
    
    extrema = []
    
    for file_entry in files_data:
        freq = file_entry['freq']
        values = file_entry['freq_data']
        fname = file_entry['name']
        
        mask = (freq >= range_min_hz) & (freq <= range_max_hz)
        if not np.any(mask):
            continue
        
        freq_masked = freq[mask]
        values_masked = values[mask]
        
        if find_maxima:
            max_idx = np.argmax(values_masked)
            extrema.append({
                'type': 'max',
                'freq': freq_masked[max_idx],
                'value': values_masked[max_idx],
                'param': selected_param,
                'file': fname,
                'domain': 'freq_raw'
            })
        
        if find_minima:
            min_idx = np.argmin(values_masked)
            extrema.append({
                'type': 'min',
                'freq': freq_masked[min_idx],
                'value': values_masked[min_idx],
                'param': selected_param,
                'file': fname,
                'domain': 'freq_raw'
            })
    
    return extrema


def find_gated_extrema(gated_data, freq_range_ghz, selected_param,
                       find_minima=True, find_maxima=True):
    """
    Find extrema in gated frequency domain data.
    
    Args:
        gated_data: list from apply_time_gate()
        freq_range_ghz: tuple (min_ghz, max_ghz)
        selected_param: str
        find_minima: bool
        find_maxima: bool
    
    Returns:
        list of dicts (same format as find_time_extrema with domain='freq_gated')
    """
    range_min_hz = freq_range_ghz[0] * 1e9
    range_max_hz = freq_range_ghz[1] * 1e9
    
    extrema = []
    
    for file_entry in gated_data:
        freq = file_entry['freq']
        values = file_entry['gated_data']
        fname = file_entry['name']
        
        mask = (freq >= range_min_hz) & (freq <= range_max_hz)
        if not np.any(mask):
            continue
        
        freq_masked = freq[mask]
        values_masked = values[mask]
        
        if find_maxima:
            max_idx = np.argmax(values_masked)
            extrema.append({
                'type': 'max',
                'freq': freq_masked[max_idx],
                'value': values_masked[max_idx],
                'param': selected_param,
                'file': fname,
                'domain': 'freq_gated'
            })
        
        if find_minima:
            min_idx = np.argmin(values_masked)
            extrema.append({
                'type': 'min',
                'freq': freq_masked[min_idx],
                'value': values_masked[min_idx],
                'param': selected_param,
                'file': fname,
                'domain': 'freq_gated'
            })
    
    return extrema


def format_time_text(extrema_list, freq_range_ghz, use_db=True):
    """
    Format time domain extrema as text.
    
    Args:
        extrema_list: combined list from find_time_extrema and find_gated_extrema
        freq_range_ghz: tuple (min_ghz, max_ghz)
        use_db: bool, if True show dB units, else magnitude
    
    Returns:
        str: formatted text
    """
    if not extrema_list:
        return ""
    
    unit = "dB" if use_db else "mag"
    lines = []
    lines.append("=" * 40)
    lines.append(f"EXTREMA (Range: {freq_range_ghz[0]:.2f}-{freq_range_ghz[1]:.2f} GHz)")
    lines.append("-" * 40)
    
    extrema_list.sort(key=lambda x: (x['domain'], x['type'], x['freq']))
    
    for ext in extrema_list:
        type_str = "MAX" if ext['type'] == 'max' else "MIN"
        freq_ghz = ext['freq'] / 1e9
        domain_str = "raw" if ext['domain'] == 'freq_raw' else "gated"
        param_str = f"{ext['param'].upper()}_{domain_str}"
        lines.append(f"{type_str} | {ext['file']} | {param_str} | {freq_ghz:.4f} GHz | {ext['value']:.2f} {unit}")
    
    return "\n".join(lines)
