import numpy as np

from analysis_field import load_field_csv, compute_layer_stats, compute_global_stats


def test_load_field_csv(tmp_path):
    p = tmp_path / "field.csv"
    p.write_text(
        "# nx = 3 ny = 2\n"
        "# layer z=0.0\n"
        "y/x 0 1 2\n"
        "0 1 2 3\n"
        "1 4 5 6\n"
        "# layer z=1.0\n"
        "y/x 0 1 2\n"
        "0 7 8 9\n"
        "1 10 11 12\n"
    )
    layers, meta = load_field_csv(str(p))
    assert layers.shape == (2, 2, 3)
    np.testing.assert_array_equal(layers[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(layers[1], [[7, 8, 9], [10, 11, 12]])
    assert meta["z_values"] == [0.0, 1.0]


def test_layer_stats():
    st = compute_layer_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert st["mean"] == 2.5 and st["min"] == 1 and st["max"] == 4 and st["max_diff"] == 3


def test_global_stats():
    g = compute_global_stats(np.array([[[-1.0, 2.0]], [[3.0, -4.0]]]))
    assert g["min"] == -4 and g["max"] == 3 and g["abs_max"] == 4 and g["range"] == 7
