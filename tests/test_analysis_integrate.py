import numpy as np

from analysis_integrate import compute_sparams, format_integration_text


def _entry(name, value):
    freq = np.linspace(1e9, 2e9, 11)
    return {"name": name, "networks": {"s11": (freq, np.full(11, value))}}


def test_integral_of_constant():
    # CubicSpline over indices 0..10 of a constant c integrates to c*(10-0)
    out = compute_sparams([_entry("a", 5.0)], ["s11"], "db", "name", True)
    assert np.isclose(out["results"]["a"]["s11"], 5.0 * 10, rtol=1e-6)


def test_sort_by_total_descending():
    out = compute_sparams([_entry("lo", 1.0), _entry("hi", 9.0)],
                          ["s11"], "db", "total", ascending=False)
    assert list(out["results"].keys()) == ["hi", "lo"]


def test_format_runs():
    out = compute_sparams([_entry("a", 5.0)], ["s11"], "db", "name", True)
    assert "Integration Results" in format_integration_text(out, 1.0, 2.0)
