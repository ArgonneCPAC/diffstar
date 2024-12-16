"""
"""

import numpy as np
from diffmah import mah_singlehalo
from diffmah.defaults import DEFAULT_MAH_PARAMS

from ...defaults import DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, FB, LGT0
from ...utils import _jax_get_dt_array
from ..fitting_kernels import (
    calculate_histories,
    calculate_histories_vmap,
    calculate_sm_sfr_fstar_history_from_mah,
    calculate_sm_sfr_history_from_mah,
)

DEFAULT_LOGM0 = 12.0


def test_calculate_sm_sfr_fstar_history_from_mah():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)

    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    fstar_indx_high = fstar_indx_high[_mask]

    dmhdt, log_mah = mah_singlehalo(DEFAULT_MAH_PARAMS, tarr, LGT0)
    args = (
        lgtarr,
        dtarr,
        dmhdt,
        log_mah,
        DEFAULT_U_MS_PARAMS,
        DEFAULT_U_Q_PARAMS,
        fstar_tdelay,
        FB,
    )
    _res = calculate_sm_sfr_fstar_history_from_mah(*args)
    for x in _res:
        assert np.all(np.isfinite(x))
    mstar, sfr, fstar = _res
    assert np.all(sfr < 1e6)
    assert np.all(mstar > 0)
    assert mstar.shape == (n_t,)
    assert sfr.shape == (n_t,)
    assert fstar.shape == (n_t,)


def test_calculate_sm_sfr_history_from_mah():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)
    dmhdt, log_mah = mah_singlehalo(DEFAULT_MAH_PARAMS, tarr, LGT0)

    args = lgtarr, dtarr, dmhdt, log_mah, DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, FB
    _res = calculate_sm_sfr_history_from_mah(*args)
    for x in _res:
        assert np.all(np.isfinite(x))
    mstar, sfr = _res
    assert np.all(sfr < 1e6)
    assert np.all(mstar > 0)
    assert mstar.shape == (n_t,)
    assert sfr.shape == (n_t,)


def test_calculate_histories():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)

    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    fstar_indx_high = fstar_indx_high[_mask]

    args = (
        lgtarr,
        dtarr,
        DEFAULT_MAH_PARAMS,
        DEFAULT_U_MS_PARAMS,
        DEFAULT_U_Q_PARAMS,
        fstar_tdelay,
        FB,
    )
    _res = calculate_histories(*args)
    for x in _res:
        assert np.all(np.isfinite(x))
    mstar, sfr, fstar, dmhdt, log_mah = _res
    assert np.all(sfr < 1e6)
    assert np.any(mstar > 1e7)
    for x in (mstar, sfr, dmhdt, log_mah):
        assert x.shape == (n_t,)
    assert np.all(np.diff(log_mah) > 0)


def test_calculate_histories_vmap():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)
    u_ms_params = np.array(DEFAULT_U_MS_PARAMS).reshape((1, -1))
    u_q_params = np.array(DEFAULT_U_Q_PARAMS).reshape((1, -1))
    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    fstar_indx_high = fstar_indx_high[_mask]

    mah_params = DEFAULT_MAH_PARAMS._make([np.zeros(1) + x for x in DEFAULT_MAH_PARAMS])
    # in_axes = (None, None, 0, 0, 0, None, None, None)
    args = (
        lgtarr,
        dtarr,
        mah_params,
        u_ms_params,
        u_q_params,
        fstar_tdelay,
        FB,
    )
    _res = calculate_histories_vmap(*args)
    for x in _res:
        assert np.all(np.isfinite(x))
    mstar_galpop, sfr_galpop, fstar_galpop, dmhdt_galpop, log_mah_galpop = _res
    for x in mstar_galpop, sfr_galpop, dmhdt_galpop, log_mah_galpop:
        assert x.shape == (1, n_t)
