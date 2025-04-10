""" """

import os

import numpy as np

from ..load_smah_data import load_smdpl_diffmah_fits

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_load_fit_mah_tpeak():
    basename = "subvol_000_diffmah_fits.h5"
    data_drn = os.path.join(_THIS_DRNAME, "testing_data")
    _res = load_smdpl_diffmah_fits(basename, data_drn=data_drn)
    for x in _res:
        assert np.all(np.isfinite(x))
    mah_params, logmp0, loss, n_points_per_fit = _res
    n_halos = mah_params.logm0.size

    assert logmp0.shape == (n_halos,)
    assert np.all(logmp0 > 10)
    assert np.all(logmp0 < 16)

    assert mah_params.t_peak.shape == (n_halos,)
    assert np.all(mah_params.t_peak > 0)
    assert np.all(mah_params.t_peak < 14)
