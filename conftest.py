# repo-root conftest: puts the project modules on sys.path and holds shared fixtures.
import numpy as np
import skrf as rf
import pytest


@pytest.fixture(scope="session")
def sample_network():
    # deterministic 2-port, each S-param a distinct shape so tests can tell them apart
    f = np.linspace(0.8e9, 2.0e9, 64)
    x = (f - f[0]) / (f[-1] - f[0])
    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 0, 0] = 10 ** ((-15 - 5 * np.cos(2 * np.pi * x)) / 20)
    s[:, 1, 0] = 10 ** ((-25 + 3 * np.sin(2 * np.pi * x)) / 20)
    s[:, 0, 1] = s[:, 1, 0]
    s[:, 1, 1] = 10 ** (-30 / 20)
    return rf.Network(f=f, s=s, z0=50, f_unit="Hz")


@pytest.fixture(scope="session")
def sample_s2p(sample_network, tmp_path_factory):
    base = tmp_path_factory.mktemp("data") / "sample_1ml"
    sample_network.write_touchstone(str(base), form="db", write_z0=False)
    return base.with_suffix(".s2p")


class FakeVar:
    """stand-in for tk.BooleanVar so backend tests dont need a display."""
    def __init__(self, value=True):
        self._v = value
    def get(self):
        return self._v


@pytest.fixture
def files_list(sample_s2p):
    return [(FakeVar(True), str(sample_s2p), {})]


@pytest.fixture(scope="session")
def family_s2p(sample_network, tmp_path_factory):
    # small viridis-orderable family named _0ml.._3ml, a small per-index S11 dB offset
    d = tmp_path_factory.mktemp("family")
    paths = []
    for i in range(4):
        s = sample_network.s.copy()
        s[:, 0, 0] = s[:, 0, 0] * 10 ** ((i * 0.5) / 20)
        ntw = rf.Network(f=sample_network.f, s=s, z0=50, f_unit="Hz")
        base = d / f"phantom_{i}ml"
        ntw.write_touchstone(str(base), form="db", write_z0=False)
        paths.append(base.with_suffix(".s2p"))
    return paths


@pytest.fixture
def family_files(family_s2p):
    return [(FakeVar(True), str(p), {}) for p in family_s2p]


@pytest.fixture(scope="session")
def tk_root():
    tk = pytest.importorskip("tkinter")
    import matplotlib
    matplotlib.use("TkAgg")
    try:
        root = tk.Tk()
    except tk.TclError as e:               # no display / no X -> skip the smoke layer
        pytest.skip(f"no display for tkinter: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture(scope="session")
def comma_s2p(sample_s2p, tmp_path_factory):
    # same data with commas for decimals (and the comment lines left alone), to exercise
    # loadFile's comma->dot path and its comment-skip
    lines = []
    for ln in sample_s2p.read_text().splitlines():
        lines.append(ln if ln.lstrip().startswith(("!", "#")) else ln.replace(".", ","))
    p = tmp_path_factory.mktemp("data_comma") / "sample_2ml.s2p"
    p.write_text("\n".join(lines))
    return p
