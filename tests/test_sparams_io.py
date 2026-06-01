import tempfile

import numpy as np

from sparams_io import loadFile


def test_loadfile_reads_touchstone(sample_s2p, sample_network):
    n = loadFile(str(sample_s2p))
    assert n.nports == 2
    assert n.frequency.npoints == sample_network.frequency.npoints
    np.testing.assert_allclose(n.f, sample_network.f, rtol=1e-9)
    # recovered to the printed dB precision of the touchstone
    np.testing.assert_allclose(n.s11.s_db.flatten(),
                               sample_network.s11.s_db.flatten(), atol=1e-3)
    np.testing.assert_allclose(n.s21.s_db.flatten(),
                               sample_network.s21.s_db.flatten(), atol=1e-3)


def test_loadfile_handles_comma_decimals(comma_s2p, sample_network):
    n = loadFile(str(comma_s2p))
    np.testing.assert_allclose(n.s11.s_db.flatten(),
                               sample_network.s11.s_db.flatten(), atol=1e-3)


def test_loadfile_does_not_leak_tempfiles(sample_s2p, tmp_path, monkeypatch):
    # loadFile writes a normalized copy to the temp dir to hand to skrf; it must not
    # leave that copy behind. currently it does (delete=False, never unlinked) -> this
    # is the HDD-clogging bug, so this test is RED until loadFile is fixed.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for _ in range(3):
        loadFile(str(sample_s2p))
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"loadFile leaked temp files: {[p.name for p in leftover]}"
