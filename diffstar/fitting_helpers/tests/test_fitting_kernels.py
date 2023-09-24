"""
"""
import numpy as np
from diffmah.individual_halo_assembly import (
    DEFAULT_MAH_PARAMS,
    _calc_halo_history,
    _get_early_late,
)

from ...defaults import DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, FB, LGT0
from ...kernels import get_sfh_from_mah_kern
from ...utils import _jax_get_dt_array
from ..fitting_kernels import (
    _sfr_history_from_mah,
    calculate_histories,
    calculate_histories_vmap,
    calculate_sm_sfr_fstar_history_from_mah,
    calculate_sm_sfr_history_from_mah,
)

DEFAULT_LOGM0 = 12.0


def _get_default_diffmah_args():
    mah_logtc, mah_k, mah_ue, mah_ul = list(DEFAULT_MAH_PARAMS.values())
    early_index, late_index = _get_early_late(mah_ue, mah_ul)
    k = DEFAULT_MAH_PARAMS["mah_k"]
    logmp = DEFAULT_LOGM0
    diffmah_args = [LGT0, logmp, mah_logtc, k, early_index, late_index]
    return diffmah_args


def test_calculate_sm_sfr_fstar_history_from_mah():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)

    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(tarr))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]

    dmhdt, log_mah = _calc_halo_history(lgtarr, *_get_default_diffmah_args())
    args = (
        lgtarr,
        dtarr,
        dmhdt,
        log_mah,
        DEFAULT_U_MS_PARAMS,
        DEFAULT_U_Q_PARAMS,
        index_select,
        fstar_indx_high,
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
    assert fstar.size < sfr.size


def test_calculate_sm_sfr_history_from_mah():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)
    dmhdt, log_mah = _calc_halo_history(lgtarr, *_get_default_diffmah_args())

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
    all_diffmah_args = _get_default_diffmah_args()

    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(tarr))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]

    args = (
        lgtarr,
        dtarr,
        all_diffmah_args,
        DEFAULT_U_MS_PARAMS,
        DEFAULT_U_Q_PARAMS,
        index_select,
        fstar_indx_high,
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

    logmp = all_diffmah_args[1]
    assert log_mah[-1] == logmp


def test_calculate_histories_vmap():
    n_t = 100
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)
    all_diffmah_args = np.array(_get_default_diffmah_args()).reshape((1, -1))
    u_ms_params = np.array(DEFAULT_U_MS_PARAMS).reshape((1, -1))
    u_q_params = np.array(DEFAULT_U_Q_PARAMS).reshape((1, -1))
    fstar_tdelay = 0.5  # gyr
    fstar_indx_high = np.searchsorted(tarr, tarr - fstar_tdelay)
    _mask = tarr > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(tarr))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]

    # in_axes = (None, None, 0, 0, 0, None, None, None)
    args = (
        lgtarr,
        dtarr,
        all_diffmah_args,
        u_ms_params,
        u_q_params,
        index_select,
        fstar_indx_high,
        fstar_tdelay,
        FB,
    )
    _res = calculate_histories_vmap(*args)
    for x in _res:
        assert np.all(np.isfinite(x))
    mstar_galpop, sfr_galpop, fstar_galpop, dmhdt_galpop, log_mah_galpop = _res
    for x in mstar_galpop, sfr_galpop, dmhdt_galpop, log_mah_galpop:
        assert x.shape == (1, n_t)


def test_sfr_history_from_mah():
    n_t = 200
    tarr = np.linspace(0.1, 10**LGT0, n_t)
    lgtarr = np.log10(tarr)
    dtarr = _jax_get_dt_array(tarr)
    all_diffmah_args = _get_default_diffmah_args()
    dmhdt, log_mah = _calc_halo_history(lgtarr, *all_diffmah_args)
    args = lgtarr, dtarr, dmhdt, log_mah, DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, FB
    sfh_from_fitting_kernels = _sfr_history_from_mah(*args)
    lgt0, logmp, logtc, k, early_index, late_index = all_diffmah_args
    mah_params = logmp, logtc, early_index, late_index

    sfh_kern = get_sfh_from_mah_kern(tobs_loop="vmap")
    sfh_from_diffstar_kernels = sfh_kern(
        tarr, mah_params, DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, LGT0, FB
    )
    assert np.allclose(sfh_from_fitting_kernels, sfh_from_diffstar_kernels, atol=0.01)
