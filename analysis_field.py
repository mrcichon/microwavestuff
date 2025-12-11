import numpy as np
import re

def parse_header(filepath):
    meta = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'nx' in line and '=' in line:
                match = re.search(r'nx\s*=\s*(\d+)', line)
                if match:
                    meta['nx'] = int(match.group(1))
            if 'ny' in line and '=' in line:
                match = re.search(r'ny\s*=\s*(\d+)', line)
                if match:
                    meta['ny'] = int(match.group(1))
            if 'nz' in line and '=' in line:
                match = re.search(r'nz\s*=\s*(\d+)', line)
                if match:
                    meta['nz'] = int(match.group(1))
            if 'x_min' in line:
                match = re.search(r'x_min\s*=\s*([\d.]+)', line)
                if match:
                    meta['x_min'] = float(match.group(1))
            if 'x_max' in line:
                match = re.search(r'x_max\s*=\s*([\d.]+)', line)
                if match:
                    meta['x_max'] = float(match.group(1))
            if 'y_min' in line:
                match = re.search(r'y_min\s*=\s*([\d.]+)', line)
                if match:
                    meta['y_min'] = float(match.group(1))
            if 'y_max' in line:
                match = re.search(r'y_max\s*=\s*([\d.]+)', line)
                if match:
                    meta['y_max'] = float(match.group(1))
            if 'z_min' in line:
                match = re.search(r'z_min\s*=\s*([\d.]+)', line)
                if match:
                    meta['z_min'] = float(match.group(1))
            if 'z_max' in line:
                match = re.search(r'z_max\s*=\s*([\d.]+)', line)
                if match:
                    meta['z_max'] = float(match.group(1))
    return meta

def load_field_csv(filepath, rows_per_layer=None, cols=None):
    meta = parse_header(filepath)
    
    if rows_per_layer is None:
        rows_per_layer = meta.get('ny')
    if cols is None:
        cols = meta.get('nx')
    
    if rows_per_layer is None or cols is None:
        raise ValueError("Could not detect grid dimensions from header and none provided")
    
    layers = []
    current_layer = []
    x_labels = None
    y_labels = []
    layer_z_values = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().replace('\r', '')
            if not line:
                continue
            
            if line.startswith('# layer'):
                if current_layer:
                    layers.append(np.array(current_layer))
                    current_layer = []
                match = re.search(r'z\s*=\s*([\d.]+)', line)
                if match:
                    layer_z_values.append(float(match.group(1)))
                continue
            
            if line.startswith('#'):
                continue
            
            if line.startswith('y/x') or line.startswith('y\\x'):
                parts = re.split(r'[,;\t\s]+', line)
                x_labels = [float(x) for x in parts[1:] if x.strip()]
                continue
            
            parts = re.split(r'[,;\t\s]+', line)
            if len(parts) < 2:
                continue
            
            try:
                y_val = float(parts[0])
                row_data = [float(x) for x in parts[1:cols+1]]
                if len(row_data) == cols:
                    current_layer.append(row_data)
                    if len(layers) == 0:
                        y_labels.append(y_val)
            except ValueError:
                continue
    
    if current_layer:
        layers.append(np.array(current_layer))
    
    result = np.array(layers)
    
    meta['x_labels'] = x_labels
    meta['y_labels'] = y_labels
    meta['z_values'] = layer_z_values
    
    return result, meta

def compute_layer_stats(layer):
    return {
        'mean': np.mean(layer),
        'variance': np.var(layer),
        'max_diff': np.max(layer) - np.min(layer),
        'min': np.min(layer),
        'max': np.max(layer),
    }

def compute_global_stats(layers):
    return {
        'min': np.min(layers),
        'max': np.max(layers),
        'abs_max': np.max(np.abs(layers)),
        'range': np.max(layers) - np.min(layers),
    }
