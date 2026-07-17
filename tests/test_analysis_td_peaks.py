import numpy as np

from analysis_td_peaks import find_td_peaks, check_peak_condition, format_td_analysis_text


def test_check_peak_condition():
    assert check_peak_condition(2.0, 1.0) is True      # |2 - 2*1| = 0
    assert check_peak_condition(2.0, 1.5) is False     # |2 - 3| = 1 > 0.5
    assert check_peak_condition(None, 1.0) is None


def test_find_td_peaks_locates_maxima_and_condition():
    t = np.linspace(0, 40, 400)
    s11 = np.exp(-((t - 10.0) ** 2) / 2)   # peak at 10 ns
    s21 = np.exp(-((t - 5.0) ** 2) / 2)    # peak at 5 ns -> 2*5 == 10 -> condition holds
    fd = [{"name": "x",
           "s11_time": {"t_ns": t, "values": s11},
           "s21_time": {"t_ns": t, "values": s21}}]
    r = find_td_peaks(fd, time_limit_ns=40)[0]
    assert abs(r["s11_max_time"] - 10) < 0.2
    assert abs(r["s21_max_time"] - 5) < 0.2
    assert r["condition_met"]                  # numpy bool, so truthy not `is True`


def test_find_td_peaks_oneport_has_no_condition():
    t = np.linspace(0, 40, 200)
    fd = [{"name": "x", "s11_time": {"t_ns": t, "values": np.exp(-((t - 8) ** 2) / 2)}}]
    r = find_td_peaks(fd, time_limit_ns=40)[0]
    assert r["s21_max_time"] is None
    assert r["condition_met"] is None
    txt = format_td_analysis_text([r], 40)
    assert "Not available" in txt and "PASS" not in txt   # 1-port: no pass/fail verdict
