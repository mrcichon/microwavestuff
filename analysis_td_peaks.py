import numpy as np

def find_td_peaks(files_data, time_limit_ns=50.0):
    """
    Find time domain peaks for S11 and S21.
    
    Args:
        files_data: list of dicts with time domain data
        time_limit_ns: time window limit in nanoseconds
    
    Returns:
        list of analysis results
    """
    results = []
    
    for file_data in files_data:
        name = file_data['name']
        
        # S11 time domain data
        t_ns = file_data['s11_time']['t_ns']
        s11_td = file_data['s11_time']['values']
        
        # Apply time limit
        time_mask = (t_ns >= 0) & (t_ns <= time_limit_ns)
        t_limited = t_ns[time_mask]
        s11_limited = s11_td[time_mask]
        
        # Find S11 max
        s11_max_idx = np.argmax(s11_limited)
        s11_max_time = t_limited[s11_max_idx]
        s11_max_val = s11_limited[s11_max_idx]
        
        result = {
            'file': name,
            's11_max_time': s11_max_time,
            's11_max_val': s11_max_val,
            's21_max_time': None,
            's21_max_val': None,
            'condition_met': None
        }
        
        # Check for S21 if available
        if 's21_time' in file_data and file_data['s21_time'] is not None:
            s21_td = file_data['s21_time']['values']
            s21_limited = s21_td[time_mask]
            
            # Find S21 max
            s21_max_idx = np.argmax(s21_limited)
            s21_max_time = t_limited[s21_max_idx]
            s21_max_val = s21_limited[s21_max_idx]
            
            result['s21_max_time'] = s21_max_time
            result['s21_max_val'] = s21_max_val
            
            # Check condition: |t_max(S11) - 2*t_max(S21)| <= 0.5
            condition_met = abs(s11_max_time - 2*s21_max_time) <= 0.5
            result['condition_met'] = condition_met
        
        results.append(result)
    
    return results


def check_peak_condition(s11_max_time, s21_max_time, tolerance=0.5):
    """Check if t_max(S11) ≈ 2 * t_max(S21)."""
    if s11_max_time is None or s21_max_time is None:
        return None
    return abs(s11_max_time - 2*s21_max_time) <= tolerance


def format_td_analysis_text(analysis_results, time_limit_ns):
    """Format TD analysis results."""
    lines = []
    lines.append("Time Domain Analysis")
    lines.append("=" * 40)
    lines.append(f"Condition: t_max(S11) == 2 * t_max(S21) (+/- 0.5ns)")
    lines.append(f"Time window: 0 - {time_limit_ns} ns")
    lines.append("-" * 40)
    lines.append("")
    
    for res in analysis_results:
        lines.append(f"{res['file']}:")
        lines.append(f"  S11 max: {res['s11_max_time']:.3f} ns @ {res['s11_max_val']:.2f} dB")
        
        if res['s21_max_time'] is not None:
            lines.append(f"  S21 max: {res['s21_max_time']:.3f} ns @ {res['s21_max_val']:.2f} dB")
            lines.append(f"  2 * t_max(S21) = {2*res['s21_max_time']:.3f} ns")
            
            if res['condition_met']:
                lines.append(f"  Condition: PASS (GREEN)")
            else:
                lines.append(f"  Condition: FAIL (RED)")
        else:
            lines.append(f"  S21: Not available (1-port device)")
        
        lines.append("")
    
    # Add summary
    valid_results = [r for r in analysis_results if r['condition_met'] is not None]
    if valid_results:
        passed = sum(1 for r in valid_results if r['condition_met'])
        total = len(valid_results)
        lines.append("-" * 40)
        lines.append(f"Summary: {passed}/{total} files pass condition")
        lines.append(f"Pass rate: {100*passed/total:.1f}%")
    
    return "\n".join(lines)
