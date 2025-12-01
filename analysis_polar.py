import numpy as np
import pandas as pd

def extract_theta_phi_series(df, tol=2.0):
    """Extract theta/phi series with automatic phi detection and stitching"""
    
    phi_vals = df["phi_deg"].unique()
    
    if len(phi_vals) == 1:
        theta = df["theta_deg"].to_numpy()
        db = df["dir_dbi"].to_numpy()
        order = np.argsort(theta)
        theta, db = theta[order], db[order]
        used = f"phi={phi_vals[0]:.0f}"
    elif len(phi_vals) == 2:
        phi_a, phi_b = sorted(phi_vals)
        if abs(phi_b - phi_a - 180) < tol:
            df_a = df[df["phi_deg"] == phi_a].copy()
            df_b = df[df["phi_deg"] == phi_b].copy()
            
            theta_a = df_a["theta_deg"].to_numpy()
            db_a = df_a["dir_dbi"].to_numpy()
            
            theta_b = 360.0 - df_b["theta_deg"].to_numpy()
            db_b = df_b["dir_dbi"].to_numpy()
            
            theta = np.concatenate([theta_a, theta_b])
            db = np.concatenate([db_a, db_b])
            order = np.argsort(theta)
            theta, db = theta[order], db[order]
            used = f"phi={phi_a:.0f}+{phi_b:.0f}"
        else:
            theta = df["theta_deg"].to_numpy()
            db = df["dir_dbi"].to_numpy()
            order = np.argsort(theta)
            theta, db = theta[order], db[order]
            used = f"phi={phi_vals}"
    else:
        theta = df["theta_deg"].to_numpy()
        db = df["dir_dbi"].to_numpy()
        order = np.argsort(theta)
        theta, db = theta[order], db[order]
        used = f"phi={len(phi_vals)} vals"
    
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
