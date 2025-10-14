import numpy as np
from scipy.interpolate import CubicSpline

def compute_sparams(files_data, params, scale_type, sort_by, ascending):
    results = {}
    
    for entry in files_data:
        name = entry['name']
        results[name] = {}
        
        for param in params:
            if param not in entry['networks']:
                continue
            
            freq, values = entry['networks'][param]
            
            if scale_type == 'linear':
                pass
            
            indices = np.arange(len(values))
            cs = CubicSpline(indices, values)
            integral = cs.integrate(indices[0], indices[-1])
            
            results[name][param] = integral
    
    if sort_by == 'name':
        sorted_items = sorted(results.items(), key=lambda x: x[0], reverse=not ascending)
    elif sort_by == 'total':
        sorted_items = sorted(results.items(), 
                            key=lambda x: sum(x[1].values()), 
                            reverse=not ascending)
    else:
        sorted_items = sorted(results.items(), 
                            key=lambda x: x[1].get(sort_by, 0), 
                            reverse=not ascending)
    
    return {
        'results': dict(sorted_items),
        'params': params,
        'scale_type': scale_type,
        'sort_by': sort_by
    }

def format_integration_text(result_dict, fmin, fmax):
    lines = []
    lines.append("Integration Results")
    lines.append(f"Scale: {result_dict['scale_type']}")
    lines.append(f"Frequency range: {fmin}-{fmax} GHz")
    lines.append(f"Sort: {result_dict['sort_by']}")
    lines.append("-" * 40)
    
    for fname, params in result_dict['results'].items():
        lines.append(f"\n{fname}:")
        total = sum(params.values())
        for param, value in params.items():
            lines.append(f"  {param.upper()}: {value:.3e}")
        if len(params) > 1:
            lines.append(f"  TOTAL: {total:.3e}")
    
    return "\n".join(lines)
