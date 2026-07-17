from analysis_overlap import parse_frequency_file, find_overlaps, format_overlap_text


def test_find_overlaps():
    assert find_overlaps([(1, 3)], [(2, 5)]) == [(2, 3)]
    assert find_overlaps([(1, 2)], [(3, 4)]) == []          # disjoint


def test_parse_frequency_file_hz_to_ghz(tmp_path):
    p = tmp_path / "ranges.txt"
    p.write_text("1000000000 2000000000\n3000000000 4000000000\n")
    assert parse_frequency_file(str(p)) == [(1.0, 2.0), (3.0, 4.0)]


def test_format_overlap_text():
    data = {"a": [(1.0, 3.0)], "b": [(2.0, 5.0)]}
    txt = format_overlap_text(data)
    assert "a <-> b" in txt and "(2.0, 3.0)" in txt
