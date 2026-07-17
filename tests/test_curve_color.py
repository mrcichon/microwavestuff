from sparams_io import PALETTE, color_for, curve_color


def test_color_for_cycles_palette():
    assert color_for(0) == PALETTE[0]
    assert color_for(len(PALETTE)) == PALETTE[0]
    assert color_for(3) == color_for(3 + len(PALETTE))


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
