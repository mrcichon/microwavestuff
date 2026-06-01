import sys
import numpy as np

def extract_time_data(files_list, freq_range_str, selected_param, use_db=True):
    """Load each selected file's freq- and time-domain data for one S-param."""
    from sparams_io import get_cached_network, display_name

    result = []

    for v, p, d in files_list:
        if not v.get():
            continue

        ntw = get_cached_network(p, d, freq_range_str)
        if ntw is None:
            continue

        try:
            s_param = getattr(ntw, selected_param)
            result.append({
                'name': display_name(p, d),
                'path': p,
                'freq': ntw.f,
                'freq_data': s_param.s_db.flatten() if use_db else s_param.s_mag.flatten(),
                't_ns': s_param.frequency.t_ns,
                'time_data': s_param.s_time_db.flatten() if use_db else s_param.s_time_mag.flatten(),
                'network': ntw,
                'color': d.get('line_color'),
                'linewidth': d.get('line_width', 1.0)
            })
        except Exception as e:
            print(f"time: {display_name(p, d)} skipped: {e}", file=sys.stderr)

    return result


def apply_time_gate(files_data, selected_param, center_ns, span_ns, use_db=True):
    """Time-gate each file's S-param and return the gated frequency response."""
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
    """Find raw freq-domain min/max for each file within the GHz range."""
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
    """Find gated freq-domain min/max for each file within the GHz range."""
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
    """Format the time/gated extrema list as text."""
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
