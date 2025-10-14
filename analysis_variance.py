import numpy as np

def compute_variance(files_data, params_list, component_types, detrend_phase=False):
    """
    Compute variance analysis across files.
    
    Args:
        files_data: list of dicts with 'name', 'networks', etc
        params_list: list of parameter names like ['s11_mag', 's11_phase']
        component_types: list of 'mag' or 'phase' for each param
        detrend_phase: whether to detrend phase data
    
    Returns:
        dict with variance analysis results
    """
    if len(files_data) < 2:
        return None
    
    n_files = len(files_data)
    freq_ref = files_data[0]['freq']
    n_frequencies = len(freq_ref)
    n_params = len(params_list)
    
    # Collect raw data
    raw_data_by_param = {param: [] for param in params_list}
    
    for file_data in files_data:
        for param in params_list:
            if '_mag' in param:
                sparam_name = param.replace('_mag', '')
                values = file_data['networks'][sparam_name][1]  # Get dB values
            else:  # phase
                sparam_name = param.replace('_phase', '')
                values = np.degrees(np.angle(file_data['networks'][sparam_name][1]))
                if np.any(np.abs(np.diff(values)) > 180):
                    values = np.degrees(np.unwrap(np.angle(file_data['networks'][sparam_name][1])))
            
            raw_data_by_param[param].append(values)
    
    # Normalize data
    normalized_data = np.zeros((n_files, n_frequencies, n_params))
    
    for i, (param, comp_type) in enumerate(zip(params_list, component_types)):
        param_data = np.array(raw_data_by_param[param])
        normalized = normalize_for_variance(param_data, is_phase=(comp_type == 'phase'), detrend=detrend_phase)
        normalized_data[:, :, i] = normalized
    
    # Calculate variance
    variance_by_param = np.var(normalized_data, axis=0, ddof=1)
    total_variance = np.sum(variance_by_param, axis=1)
    
    # Calculate contribution percentage
    variance_contribution = np.zeros_like(variance_by_param)
    for freq_idx in range(n_frequencies):
        if total_variance[freq_idx] > 1e-10:
            variance_contribution[freq_idx] = (variance_by_param[freq_idx] / 
                                             total_variance[freq_idx]) * 100
    
    mean_by_param = np.mean(normalized_data, axis=0)
    std_by_param = np.std(normalized_data, axis=0, ddof=1)
    
    norm_info = 'Phase: detrended+z-score, Magnitude: z-score' if detrend_phase else 'Phase: z-score, Magnitude: z-score'
    
    return {
        'frequencies': freq_ref,
        'variance_by_param': variance_by_param,
        'variance_contribution': variance_contribution,
        'total_variance': total_variance,
        'mean_by_param': mean_by_param,
        'std_by_param': std_by_param,
        'n_params': n_params,
        'param_list': params_list,
        'file_count': n_files,
        'normalized_data': normalized_data,
        'normalization_info': norm_info
    }


def normalize_for_variance(data, is_phase=False, detrend=False):
    """Normalize data for variance calculation."""
    if is_phase and detrend:
        n_samples, n_frequencies = data.shape
        detrended = np.zeros_like(data)
        
        for i in range(n_samples):
            x = np.arange(n_frequencies)
            coeffs = np.polyfit(x, data[i], 1)
            trend = np.polyval(coeffs, x)
            detrended[i] = data[i] - trend
        
        mean = np.mean(detrended)
        std = np.std(detrended) + 1e-10
        return (detrended - mean) / std
    else:
        mean = np.mean(data)
        std = np.std(data) + 1e-10
        return (data - mean) / std


def format_variance_text(variance_data):
    """Format variance analysis results."""
    if variance_data is None:
        return ""
    
    lines = []
    mean_var = np.mean(variance_data['total_variance'])
    std_var = np.std(variance_data['total_variance'])
    
    lines.append(f"Variance Statistics (Normalized Data):")
    lines.append(f"Normalization: {variance_data['normalization_info']}")
    lines.append(f"Mean total variance: {mean_var:.6f}")
    lines.append(f"Std of variance: {std_var:.6f}")
    lines.append(f"Files analyzed: {variance_data['file_count']}")
    
    return "\n".join(lines)
