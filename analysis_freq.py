import numpy as np
from pathlib import Path

def extract_freq_data(files_list, freq_range_str, selected_params, use_db=True):
    """
    Extract frequency domain data from file list.
    
    Args:
        files_list: list of (BooleanVar, path, metadata_dict)
        freq_range_str: e.g. "0.4-4.0ghz"
        selected_params: list of str like ['s11', 's21']
        use_db: bool, if True use dB scale, else linear magnitude
    
    Returns:
        list of dicts with keys:
            'name': str
            'path': str
            'freq': ndarray
            'params': dict mapping param_name -> values_array
            'color': str or None
            'linewidth': float
    """
    from sparams_io import loadFile
    
    result = []
    
    for v, p, d in files_list:
        if not v.get():
            continue
        
        ext = Path(p).suffix.lower()
        loaded = False
        
        if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p', '.csv']:
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
                
                params_data = {}
                for param in selected_params:
                    s_param = getattr(ntw, param, None)
                    if s_param is None:
                        continue
                    arr = s_param.s_db.flatten() if use_db else s_param.s_mag.flatten()
                    params_data[param] = arr
                
                if params_data:
                    result.append({
                        'name': fname,
                        'path': p,
                        'freq': ntw.f,
                        'params': params_data,
                        'color': d.get('line_color'),
                        'linewidth': d.get('line_width', 1.0)
                    })
                    loaded = True
                
            except Exception:
                pass
        
        if not loaded and ext == '.csv' and 's11' in selected_params:
            try:
                import pandas as pd
                
                if 'csv_data' not in d:
                    df = pd.read_csv(p)
                    d['csv_data'] = (df.iloc[:, 0].values, df.iloc[:, 1].values)
                
                freq, arr = d['csv_data']
                fname = Path(p).stem
                
                result.append({
                    'name': fname,
                    'path': p,
                    'freq': freq,
                    'params': {'s11': arr},
                    'color': d.get('line_color'),
                    'linewidth': d.get('line_width', 1.0)
                })
                
            except Exception:
                pass
    
    return result


def find_extrema(files_data, selected_params, freq_range_ghz, find_minima=True, find_maxima=True):
    """
    Find frequency extrema for each file and parameter.
    
    Args:
        files_data: list from extract_freq_data()
        selected_params: list of param names
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
    """
    range_min_hz = freq_range_ghz[0] * 1e9
    range_max_hz = freq_range_ghz[1] * 1e9
    
    extrema = []
    
    for file_entry in files_data:
        freq = file_entry['freq']
        fname = file_entry['name']
        
        mask = (freq >= range_min_hz) & (freq <= range_max_hz)
        if not np.any(mask):
            continue
        
        freq_masked = freq[mask]
        
        for param in selected_params:
            if param not in file_entry['params']:
                continue
            
            values = file_entry['params'][param]
            values_masked = values[mask]
            
            if find_maxima:
                max_idx = np.argmax(values_masked)
                extrema.append({
                    'type': 'max',
                    'freq': freq_masked[max_idx],
                    'value': values_masked[max_idx],
                    'param': param,
                    'file': fname
                })
            
            if find_minima:
                min_idx = np.argmin(values_masked)
                extrema.append({
                    'type': 'min',
                    'freq': freq_masked[min_idx],
                    'value': values_masked[min_idx],
                    'param': param,
                    'file': fname
                })
    
    extrema.sort(key=lambda x: (x['param'], x['type'], x['freq']))
    return extrema


def format_freq_text(extrema_list, freq_range_ghz, use_db=True):
    """
    Format extrema data as text output.
    
    Args:
        extrema_list: list from find_extrema()
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
    
    for ext in extrema_list:
        type_str = "MAX" if ext['type'] == 'max' else "MIN"
        freq_ghz = ext['freq'] / 1e9
        lines.append(f"{type_str} | {ext['file']} | {ext['param'].upper()} | {freq_ghz:.4f} GHz | {ext['value']:.2f} {unit}")
    
    return "\n".join(lines)
