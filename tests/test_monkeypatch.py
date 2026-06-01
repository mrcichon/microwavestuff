import numpy as np
import pytest
import skrf as rf


def test_patch_gates_oneport():
    import skrf_patch  # noqa: F401  binds the patch on import

    f = np.linspace(1e9, 3e9, 201)
    s = (0.5 * np.exp(1j * np.linspace(0, 8 * np.pi, 201))).reshape(-1, 1, 1)
    ntw = rf.Network(f=f, s=s, z0=50, f_unit="Hz")

    gated = ntw.time_gate(center=2.0, span=1.0)
    assert gated.frequency.npoints == ntw.frequency.npoints
    assert gated.s.shape == ntw.s.shape
    assert np.isfinite(gated.s).all()
    assert not np.allclose(gated.s, ntw.s)        # gating actually altered the response


def test_importing_ui_main_installs_patch():
    # the dual-entry fix: launching via ui_main (not just mainscript) must patch too
    if rf.__version__ <= "0.17.0":
        pytest.skip("patch only applies for skrf>0.17")
    import ui_main  # noqa: F401
    assert rf.Network.time_gate.__name__ == "time_gate_v017"
