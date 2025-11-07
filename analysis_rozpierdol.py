import numpy as np
from pathlib import Path

def extract_overlay_data(files_list, freq_range_str, file_params_map, mode):
    from sparams_io import loadFile
    
    result = []
    
    for v, p, d in files_list:
        if not v.get():
            continue
        
        params = file_params_map.get(p, [])
        if not params:
            continue
        
        ext = Path(p).suffix.lower()
        
        if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
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
                
                for param in params:
                    s_param = getattr(ntw, param)
                    
                    if mode == 'db':
                        arr = s_param.s_db.flatten()
                    elif mode == 'mag':
                        arr = s_param.s_mag.flatten()
                    else:
                        arr = s_param.s_deg.flatten()
                    
                    result.append({
                        'name': fname,
                        'param': param,
                        'label': f"{fname} - {param.upper()}",
                        'freq': ntw.f,
                        'values': arr,
                        'color': d.get('line_color'),
                        'linewidth': d.get('line_width', 1.0)
                    })
                
            except Exception:
                pass
    
    return result

def format_overlay_text(data, mode):
    if not data:
        return ""
    
    if mode == 'db':
        unit = 'dB'
    elif mode == 'mag':
        unit = 'mag'
    else:
        unit = 'deg'
    
    lines = []
    lines.append("=" * 60)
    lines.append("OVERLAY PLOT DATA")
    lines.append("-" * 60)
    
    for entry in data:
        mean_val = np.mean(entry['values'])
        min_val = np.min(entry['values'])
        max_val = np.max(entry['values'])
        lines.append(f"{entry['label']:40s} | Mean: {mean_val:8.2f} {unit} | Min: {min_val:8.2f} | Max: {max_val:8.2f}")
    
    return "\n".join(lines)
