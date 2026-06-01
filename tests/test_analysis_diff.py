import numpy as np
import skrf as rf

from analysis_diff import compute_diff, format_diff_text


def _net(s11_db, n=16):
    f = np.linspace(1e9, 2e9, n)
    s = np.zeros((n, 2, 2), dtype=complex)
    s[:, 0, 0] = 10 ** (s11_db / 20)
    s[:, 1, 0] = 10 ** ((s11_db - 5) / 20)
    return rf.Network(f=f, s=s, z0=50, f_unit="Hz")


def test_compute_diff_is_db_subtraction():
    out = compute_diff(_net(-10.0), _net(-15.0), ["s11"])
    np.testing.assert_allclose(out["freq"], _net(-10.0).f)
    np.testing.assert_allclose(out["s11"], 5.0, atol=1e-6)   # -10 - (-15)


def test_compute_diff_text_runs():
    pair = {"name1": "a", "name2": "b",
            "diff_data": compute_diff(_net(-10.0), _net(-15.0), ["s11"])}
    txt = format_diff_text([pair], ["s11"])
    assert "a - b" in txt and "Mean diff" in txt
