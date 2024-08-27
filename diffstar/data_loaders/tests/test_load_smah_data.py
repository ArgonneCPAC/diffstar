"""
"""

import os

import numpy as np

from ..load_smah_data import load_fit_mah_tpeak

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_load_fit_mah_tpeak():
    basename = "subvol_000_diffmah_fits.h5"
    data_drn = os.path.join(_THIS_DRNAME, "testing_data")
    _res = load_fit_mah_tpeak(basename, data_drn=data_drn)
    for x in _res:
        assert np.all(np.isfinite(x))
    mah_fit_params, logmp, t_peak = _res
    n_halos, n_params = mah_fit_params.shape

    assert logmp.shape == (n_halos,)
    assert np.all(logmp > 10)
    assert np.all(logmp < 16)

    assert t_peak.shape == (n_halos,)
    assert np.all(t_peak > 0)
    assert np.all(t_peak < 14)
