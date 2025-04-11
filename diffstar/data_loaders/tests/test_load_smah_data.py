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

    # test istart:iend feature
    n_halos_small = 4
    assert n_halos_small < n_halos
    _res = load_smdpl_diffmah_fits(
        basename, data_drn=data_drn, istart=0, iend=n_halos_small
    )
    mah_params_small, logmp0_small, loss_small, n_points_per_fit_small = _res
    assert logmp0_small.shape == (n_halos_small,)
    assert np.allclose(mah_params_small.logm0, mah_params.logm0[:n_halos_small])
    assert np.allclose(logmp0_small, logmp0[:n_halos_small])
    assert np.allclose(loss_small, loss[:n_halos_small])
    assert np.allclose(n_points_per_fit_small, n_points_per_fit[:n_halos_small])
