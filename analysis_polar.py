import numpy as np
import pandas as pd

def extract_theta_phi_series(df, tol=2.0):
    """Extract theta/phi series with automatic phi detection and mirroring"""
    
    theta_raw = df["theta_deg"].to_numpy()
    theta_normalized = theta_raw % 360.0
    
    df = df.copy()
    df["theta_deg"] = theta_normalized
    df = df.sort_values("theta_deg").reset_index(drop=True)
    
    theta = df["theta_deg"].to_numpy()
    db = df["dir_dbi"].to_numpy()
    
    phi_vals = df["phi_deg"].unique()
    used = f"phi={phi_vals[0]:.0f}" if len(phi_vals) == 1 else f"phi={phi_vals}"
    
    if len(theta) > 1 and abs(theta[-1] - 360.0) > 1.0:
        theta = np.append(theta, 360.0)
        db = np.append(db, db[0])
    
    return theta, db, used

def prepare_polar_series(angle_deg, values, min_db, max_db, label="", color="#000000"):
    """Prepare data for polar plotting"""
    angle_rad = np.array(angle_deg) * np.pi / 180.0
    vals = np.clip(np.array(values), min_db, max_db)
    vals_offset = vals - min_db
    return {
        "theta": angle_rad,
        "r": vals_offset,
        "label": label,
        "color": color
    }
