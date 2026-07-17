import numpy as np

from analysis_time import (extract_time_data, apply_time_gate,
                           find_time_extrema, find_gated_extrema)


def test_find_time_extrema_freq_domain():
    freq = np.linspace(1e9, 2e9, 11)
    fd = [{"name": "f", "freq": freq, "freq_data": np.array(
        [0, -1, -2, -9, -2, -1, 0, 1, 2, 3, 4.0])}]
    ext = find_time_extrema(fd, "s11", (1.0, 2.0))
    by = {e["type"]: e for e in ext}
    assert by["min"]["value"] == -9
    assert by["max"]["value"] == 4
    assert all(e["domain"] == "freq_raw" for e in ext)


def test_extract_time_data_and_gate(files_list):
    import skrf_patch  # noqa: F401  applies the skrf time_gate patch

    raw = extract_time_data(files_list, "0.8-2.0ghz", "s11", use_db=True)
    assert len(raw) == 1
    assert raw[0]["time_data"].size == raw[0]["freq"].size

    gated = apply_time_gate(raw, "s11", center_ns=2.0, span_ns=2.0, use_db=True)
    assert len(gated) == 1
    assert gated[0]["gated_data"].size == raw[0]["freq"].size

    gext = find_gated_extrema(gated, (0.8, 2.0), "s11")
    assert all(e["domain"] == "freq_gated" for e in gext)
