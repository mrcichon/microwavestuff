import numpy as np

from analysis_shape import compute_cross_correlation_matrix, compute_shape_matrix


def test_xcorr_identical_signal():
    x = np.sin(np.linspace(0, 6, 128))
    lag, val = compute_cross_correlation_matrix(x, x, normalize=True)
    assert lag == 0
    assert val > 0.99


def test_xcorr_recovers_shift():
    a = np.zeros(128); a[60] = 1.0
    b = np.zeros(128); b[65] = 1.0
    lag, _ = compute_cross_correlation_matrix(a, b, normalize=True)
    assert abs(lag) == 5


def test_shape_matrix_identical_and_l2_zero():
    f = np.linspace(1e9, 2e9, 64)
    sig = np.sin(np.linspace(0, 6, 64))
    fd = [{"name": n, "signal": sig.copy(), "freq": f} for n in "abc"]

    m = compute_shape_matrix(fd, "s21", metric="xcorr")
    np.testing.assert_allclose(np.diag(m["matrix"]), 1.0)
    assert (m["matrix"] > 0.99).all()

    m2 = compute_shape_matrix(fd, "s21", metric="l2")
    np.testing.assert_allclose(m2["matrix"], 0.0, atol=1e-6)


def test_shape_matrix_needs_two():
    fd = [{"name": "a", "signal": np.zeros(8), "freq": np.arange(8)}]
    assert compute_shape_matrix(fd, "s21") is None
