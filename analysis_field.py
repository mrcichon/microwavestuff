import numpy as np

def load_field_csv(filepath, rows_per_layer, cols):
    data = np.loadtxt(filepath, delimiter=',')
    if data.shape[1] != cols:
        raise ValueError(f"Expected {cols} columns, got {data.shape[1]}")
    if data.shape[0] % rows_per_layer != 0:
        raise ValueError(f"Row count {data.shape[0]} not divisible by {rows_per_layer}")
    n_layers = data.shape[0] // rows_per_layer
    return data.reshape(n_layers, rows_per_layer, cols)

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
