import numpy as np

def compute_diff(ntw1, ntw2, params):
    results = {}
    freq1 = ntw1.f
    freq2 = ntw2.f
    
    if not np.allclose(freq1, freq2, rtol=1e-6):
        common_min = max(freq1.min(), freq2.min())
        common_max = min(freq1.max(), freq2.max())
        mask1 = (freq1 >= common_min) & (freq1 <= common_max)
        mask2 = (freq2 >= common_min) & (freq2 <= common_max)
        freq1 = freq1[mask1]
        freq2 = freq2[mask2]
        if len(freq1) != len(freq2):
            return None
        freq = freq1
    else:
        freq = freq1
        mask1 = mask2 = slice(None)
    
    results['freq'] = freq
    
    for param in params:
        s1 = getattr(ntw1, param, None)
        s2 = getattr(ntw2, param, None)
        if s1 is None or s2 is None:
            continue
        
        db1 = s1.s_db.flatten()
        db2 = s2.s_db.flatten()
        
        if isinstance(mask1, np.ndarray):
            db1 = db1[mask1]
            db2 = db2[mask2]
        
        results[param] = db1 - db2
    
    return results

def format_diff_text(diff_pairs, params):
    if not diff_pairs:
        return ""
    
    lines = []
    lines.append("Difference Analysis (dB)")
    lines.append("=" * 50)
    
    for i, pair in enumerate(diff_pairs):
        name1, name2 = pair['name1'], pair['name2']
        diff_data = pair['diff_data']
        lines.append(f"\n[{i+1}] {name1} - {name2}")
        lines.append("-" * 40)
        
        for param in params:
            if param not in diff_data:
                continue
            vals = diff_data[param]
            lines.append(f"  {param.upper()}:")
            lines.append(f"    Mean diff: {np.mean(vals):+.3f} dB")
            lines.append(f"    Std diff:  {np.std(vals):.3f} dB")
            lines.append(f"    Max diff:  {np.max(vals):+.3f} dB")
            lines.append(f"    Min diff:  {np.min(vals):+.3f} dB")
    
    return "\n".join(lines)
