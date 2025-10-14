import numpy as np
import pandas as pd

def extract_theta_phi_series(df, tol=2.0):
    """Extract theta/phi series with automatic phi detection and mirroring"""
    def nearest_mod180(phivals, p, tol=2.0):
        p2 = (p + 180.0) % 360.0
        diffs = np.minimum(np.abs(phivals - p2), 360.0 - np.abs(phivals - p2))
        idx = np.where(diffs <= tol)[0]
        return int(idx[0]) if idx.size else None
    
    phivals = np.array(sorted(np.round(df["phi_deg"].to_numpy(), 1)))
    uniq = np.unique(phivals)
    if uniq.size == 0:
        raise ValueError("No phi values in file")
    
    counts = [(phi, (phivals == phi).sum()) for phi in uniq]
    counts.sort(key=lambda x: x[1], reverse=True)
    phi_a = counts[0][0]
    j = nearest_mod180(uniq, phi_a, tol=tol)
    
    if j is not None:
        phi_b = float(uniq[j])
        a = df[np.isclose(df["phi_deg"], phi_a, atol=tol)].sort_values("theta_deg")
        b = df[np.isclose(df["phi_deg"], phi_b, atol=tol)].sort_values("theta_deg", ascending=False)
        theta = np.concatenate([a["theta_deg"].to_numpy(), (360.0 - b["theta_deg"].to_numpy())])
        db = np.concatenate([a["dir_dbi"].to_numpy(), b["dir_dbi"].to_numpy()])
        if theta.size and abs(theta[-1]-360.0) < 1e-6:
            theta, db = theta[:-1], db[:-1]
        used = f"φ≈{phi_a}° & {phi_b}°"
    else:
        a = df[np.isclose(df["phi_deg"], phi_a, atol=tol)].sort_values("theta_deg")
        theta = a["theta_deg"].to_numpy()
        db = a["dir_dbi"].to_numpy()
        span = float(theta.max() - theta.min()) if theta.size else 0.0
        if span < 300:
            theta2 = 360.0 - theta[::-1]
            db2 = db[::-1]
            if theta2.size and abs(theta2[0] - 360.0) < 1e-6:
                theta2, db2 = theta2[1:], db2[1:]
            theta = np.concatenate([theta, theta2])
            db = np.concatenate([db, db2])
        used = f"φ≈{phi_a}°"
    
    if theta.size > 0:
        if abs(theta[-1] - 360.0) > 1e-3:
            theta = np.append(theta, 360.0)
            db = np.append(db, db[0])
        elif abs(theta[0]) > 1e-3:
            theta = np.insert(theta, 0, 0.0)
            db = np.insert(db, 0, db[-1])
    
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
