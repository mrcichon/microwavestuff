import numpy as np

from analysis_variance import compute_variance, normalize_for_variance


def _files(values_per_file):
    freq = np.linspace(1e9, 2e9, 16)
    return [{"name": str(i), "freq": freq,
             "networks": {"s11": (freq, np.full(16, v))}}
            for i, v in enumerate(values_per_file)]


def test_single_param_is_full_contribution():
    out = compute_variance(_files([-10.0, -20.0]), ["s11_mag"], ["mag"], detrend_phase=False)
    assert out["file_count"] == 2
    nz = out["total_variance"] > 1e-10
    np.testing.assert_allclose(out["variance_contribution"][nz], 100.0)


def test_needs_two_files():
    assert compute_variance(_files([-10.0]), ["s11_mag"], ["mag"]) is None


def test_normalize_is_zscore():
    out = normalize_for_variance(np.array([[1.0, 2.0, 3.0, 4.0]]))
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-6
