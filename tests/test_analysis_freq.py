import numpy as np

from analysis_freq import extract_freq_data, find_extrema, format_freq_text


def test_find_extrema_locates_min_and_max():
    freq = np.linspace(1e9, 2e9, 11)
    vals = np.array([0, -1, -2, -3, -10, -3, -2, -1, 0, 1, 2.0])  # min idx4, max idx10
    files_data = [{"name": "f", "freq": freq, "params": {"s11": vals}}]
    ext = find_extrema(files_data, ["s11"], (1.0, 2.0), find_minima=True, find_maxima=True)
    by = {e["type"]: e for e in ext}
    assert by["min"]["value"] == -10 and np.isclose(by["min"]["freq"], freq[4])
    assert by["max"]["value"] == 2 and np.isclose(by["max"]["freq"], freq[10])


def test_find_extrema_respects_range():
    freq = np.linspace(1e9, 2e9, 11)
    vals = np.arange(11.0)                      # global max at 2.0 GHz
    files_data = [{"name": "f", "freq": freq, "params": {"s11": vals}}]
    ext = find_extrema(files_data, ["s11"], (1.0, 1.4), find_minima=False, find_maxima=True)
    assert ext[0]["freq"] <= 1.4e9 + 1            # max confined to the window


def test_extract_freq_data_reads_file(files_list):
    out = extract_freq_data(files_list, "0.8-2.0ghz", ["s11", "s21"], use_db=True)
    assert len(out) == 1
    assert set(out[0]["params"]) == {"s11", "s21"}
    assert out[0]["freq"][0] >= 0.8e9
    assert format_freq_text(
        find_extrema(out, ["s11"], (0.8, 2.0)), (0.8, 2.0)).startswith("=")
