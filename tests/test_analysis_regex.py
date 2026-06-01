import numpy as np

from analysis_regex import (extract_regex_value, _mono_mask, find_best_drop,
                            compute_kendall_tau, compute_max_displacement,
                            mask_to_ranges, mask_to_index_ranges,
                            compute_congruence, compute_shift_tracking)


def test_extract_regex_value():
    assert extract_regex_value("phantom_50ml", r"_(\d+)ml", 1) == 50.0
    assert extract_regex_value("no_match_here", r"_(\d+)ml", 1) is None


def test_mono_mask():
    base = np.array([0.0, 1.0, 2.0, 3.0])
    stacked = np.array([base + i for i in range(5)])      # curves in order at every freq
    assert _mono_mask(stacked, strict=False).all()
    assert _mono_mask(stacked[::-1], strict=False).all()  # reversed order is still monotonic
    # a crossing at the second frequency breaks it there only
    crossing = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 1.0]])
    np.testing.assert_array_equal(_mono_mask(crossing, strict=False), [True, False])


def test_kendall_tau_endpoints():
    ordered = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_allclose(compute_kendall_tau(ordered), [1.0, 1.0])
    np.testing.assert_allclose(compute_kendall_tau(ordered[::-1]), [-1.0, -1.0])


def test_max_displacement():
    ordered = np.array([[0.0], [1.0], [2.0]])
    assert compute_max_displacement(ordered)[0] == 0
    reversed_ = np.array([[2.0], [1.0], [0.0]])
    assert compute_max_displacement(reversed_)[0] == 2


def test_find_best_drop_removes_the_bad_curve():
    base = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    good = [base + 10 * i for i in range(4)]
    bad = np.array([100.0, -100.0, 100.0, -100.0, 100.0])   # crosses everything
    s = np.array([good[0], good[1], bad, good[2], good[3]])
    m0, _ = find_best_drop(s, 0, strict=False)
    m1, dropped = find_best_drop(s, 1, strict=False)
    assert m1.sum() > m0.sum()
    assert dropped == [2]
    assert m1.all()


def test_mask_to_ranges():
    mask = np.array([False, True, True, False, True])
    freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert mask_to_ranges(mask, freqs) == [(1.0, 2.0), (4.0, 4.0)]
    assert mask_to_index_ranges(mask) == [(1, 2), (4, 4)]


def test_congruence_separates_translates_from_noise():
    x = np.linspace(0, 3, 200)
    shape = 5 * np.sin(x)
    translates = np.array([shape + 2.0 * i for i in range(6)])   # same shape, vertical offsets
    _, mask = compute_congruence(translates, window=21, threshold=0.9, mode="vertical")
    assert mask.mean() > 0.9

    noise = np.random.default_rng(0).normal(size=(6, 200))
    score, _ = compute_congruence(noise, window=21, threshold=0.9, mode="vertical")
    assert score.mean() < 0.5


def test_shift_tracking_recovers_linear_slide():
    x = np.arange(200)
    vals = np.arange(11)
    centers = 120 - 4 * vals                                     # 4 samples per unit
    s = np.array([-10 * np.exp(-((x - c) ** 2) / 50.0) for c in centers])
    freqs = np.linspace(1e9, 3e9, 200)
    df = freqs[1] - freqs[0]
    _, region = compute_shift_tracking(s, vals, freqs, window=61, max_lag=30,
                                       shape_thresh=0.8, mono_thresh=0.9)
    assert region is not None
    np.testing.assert_allclose(region["slope_hz_per_unit"], -4 * df, rtol=0.1)
    assert region["r2"] > 0.98
