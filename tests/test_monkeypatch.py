import numpy as np
import skrf as rf


def test_time_gate_patch_runs_on_oneport():
    # importing mainscript applies the skrf>0.17 time_gate/delay patch at module load
    # (the UI only starts under __main__, so this import is safe and side-effect-light)
    import mainscript  # noqa: F401

    f = np.linspace(1e9, 3e9, 201)
    s = (0.5 * np.exp(1j * np.linspace(0, 8 * np.pi, 201))).reshape(-1, 1, 1)
    ntw = rf.Network(f=f, s=s, z0=50, f_unit="Hz")

    gated = ntw.time_gate(center=2.0, span=1.0)
    assert gated.frequency.npoints == ntw.frequency.npoints
    assert gated.s.shape == ntw.s.shape
    assert np.isfinite(gated.s).all()
    assert not np.allclose(gated.s, ntw.s)        # gating actually altered the response
