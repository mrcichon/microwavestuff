import numpy as np
import pandas as pd

from analysis_polar import prepare_polar_series, extract_theta_phi_series


def test_prepare_polar_series_clip_and_offset():
    out = prepare_polar_series([0, 90, 180, 270], [-5, -10, -50, 0],
                               min_db=-30, max_db=0)
    np.testing.assert_allclose(out["theta"], np.deg2rad([0, 90, 180, 270]))
    # clip to [-30, 0] then shift up by -min_db -> [25, 20, 0, 30]
    np.testing.assert_allclose(out["r"], [25, 20, 0, 30])


def test_extract_theta_phi_single_phi_sorts_and_closes():
    df = pd.DataFrame({"phi_deg": [0, 0, 0],
                       "theta_deg": [0, 90, 45],
                       "dir_dbi": [1.0, 2.0, 3.0]})
    theta, db, used = extract_theta_phi_series(df)
    np.testing.assert_array_equal(theta[:3], [0, 45, 90])
    assert theta[-1] == 360.0           # loop closure appended
    assert db[-1] == db[0]
    assert "phi=0" in used


def test_extract_theta_phi_two_planes_stitch():
    # phi 0 and 180 -> second plane mirrored as 360-theta and stitched
    df = pd.DataFrame({
        "phi_deg": [0, 0, 180, 180],
        "theta_deg": [0, 90, 30, 90],
        "dir_dbi": [1.0, 2.0, 3.0, 4.0],
    })
    theta, db, used = extract_theta_phi_series(df)
    assert theta.min() >= 0 and theta.max() <= 360
    assert "+" in used                  # combined-plane label
