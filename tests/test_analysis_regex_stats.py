import numpy as np

from analysis_regex_stats import (format_perfile_range, format_perfile_single,
                                  format_group_range, format_group_single)


def test_perfile_range():
    f = np.linspace(0.4e9, 3.0e9, 1002)
    a = np.full(1002, -6.0)
    b = np.full(1002, -7.0)
    c = np.full(1002, -6.0); c[500] = -10.0      # C has a clear minimum
    txt = format_perfile_range([("A", f, a), ("B", f, b), ("C", f, c)], 0.4, 3.0)
    assert txt.startswith("=== Regex Stats ===")
    assert "Range: 0.400000 .. 3.000000 GHz   (N=1002)" in txt
    assert "Metric: |S| [dB]" in txt
    assert "A: -6" in txt and "B: -7" in txt
    assert "value_min=-10" in txt                # C's minimum
    assert "0-1: 1   (A vs B)" in txt            # A,B flat: |(-6) - (-7)| = 1
    assert "Global mean |neighbor diff| in range:" in txt


def test_perfile_single():
    txt = format_perfile_single([("A", -7.0), ("B", -6.0)], 2.2)
    assert "Single frequency: 2.200000 GHz" in txt
    assert "== Value per file at selected frequency ==" in txt
    assert "Mean across files: -6.5" in txt
    assert "Std across files: 0.5" in txt
    assert "0-1: 1   (A vs B)" in txt


def test_group_range():
    f = np.linspace(1e9, 2e9, 11)
    members = [("54.1", f, np.full(11, -6.0)), ("54.2", f, np.full(11, -7.0))]
    txt = format_group_range({54.0: members}, 1.0, 2.0, r"(\d+).", 1)
    assert "=== Regex Stats Group ===" in txt
    assert "Pattern: (\\d+).   group=1" in txt
    assert "== Group 54.0 ==" in txt
    assert "Files in group: 2" in txt
    assert "Members: 54.1, 54.2" in txt
    assert "Band mean across files: mean=-6.5   std=0.5" in txt
    assert "Minimum of mean trace:" in txt
    assert "Per-file minima:" in txt


def test_group_single():
    txt = format_group_single({54.0: [("54.1", -7.0), ("54.2", -6.0)]}, 2.2, r"(\d+).", 1)
    assert "== Group 54.0 ==" in txt
    assert "Mean at selected frequency: -6.5" in txt
    assert "Std at selected frequency: 0.5" in txt
