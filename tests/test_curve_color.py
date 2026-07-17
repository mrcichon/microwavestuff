from sparams_io import PALETTE, color_for, curve_color


def test_color_for_starts_with_palette_then_stays_unique():
    assert [color_for(i) for i in range(len(PALETTE))] == PALETTE
    colors = [color_for(i) for i in range(50)]
    assert len(set(colors)) == 50


def test_curve_color_override_beats_auto():
    assert curve_color({"line_color": "#123456", "auto_color": "#abcdef"}) == "#123456"
    assert curve_color({"line_color": None, "auto_color": "#abcdef"}) == "#abcdef"
    assert curve_color({}) is None


def test_extract_freq_data_carries_auto_color(files_list):
    from analysis_freq import extract_freq_data
    v, p, d = files_list[0]
    d["auto_color"] = "#abcdef"
    out = extract_freq_data([(v, p, d)], "0.8-2.0ghz", ["s11"], use_db=True)
    assert out[0]["color"] == "#abcdef"
    d["line_color"] = "#123456"
    out = extract_freq_data([(v, p, d)], "0.8-2.0ghz", ["s11"], use_db=True)
    assert out[0]["color"] == "#123456"
