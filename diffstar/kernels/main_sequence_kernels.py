"""
"""
from collections import OrderedDict, namedtuple

import numpy as np
from diffmah.individual_halo_assembly import (
    DEFAULT_MAH_PARAMS,
    _calc_halo_history_scalar,
    _rolling_plaw_vs_logt,
)
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _jax_get_dt_array, _sigmoid
from .gas_consumption import _gas_conversion_kern

DEFAULT_MS_PDICT = OrderedDict(
    lgmcrit=12.0,
    lgy_at_mcrit=-1.0,
    indx_lo=1.0,
    indx_hi=-1.0,
    tau_dep=2.0,
)
MSParams = namedtuple("MSParams", list(DEFAULT_MS_PDICT.keys()))
DEFAULT_MS_PARAMS = MSParams(*list(DEFAULT_MS_PDICT.values()))

MS_PARAM_BOUNDS_PDICT = OrderedDict(
    lgmcrit=(9.0, 13.5),
    lgy_at_mcrit=(-3.0, 0.0),
    indx_lo=(0.0, 5.0),
    indx_hi=(-5.0, 0.0),
    tau_dep=(0.01, 20.0),
)

INDX_K = 9.0  # Main sequence efficiency transition speed.


def calculate_sigmoid_bounds(param_bounds):
    bounds_out = OrderedDict()

    for key in param_bounds:
        _bounds = (
            float(np.mean(param_bounds[key])),
            abs(float(4.0 / np.diff(param_bounds[key])[0])),
        )
        bounds_out[key] = _bounds + param_bounds[key]
    return bounds_out


MS_BOUNDING_SIGMOID_PDICT = calculate_sigmoid_bounds(MS_PARAM_BOUNDS_PDICT)


@jjit
def _lax_ms_sfh_scalar_kern(t_form, mah_params, ms_params, lgt0, fb, t_table):
    mah_k = DEFAULT_MAH_PARAMS["mah_k"]
    logmp, logtc, early, late = mah_params
    all_mah_params = lgt0, logmp, logtc, mah_k, early, late
    lgt_form = jnp.log10(t_form)
    log_mah_at_tform = _rolling_plaw_vs_logt(lgt_form, *all_mah_params)

    sfr_eff_params = ms_params[:4]
    sfr_eff = _sfr_eff_plaw(log_mah_at_tform, *sfr_eff_params)

    tau_dep = ms_params[4]
    tau_dep_max = MS_BOUNDING_SIGMOID_PDICT["tau_dep"][3]

    dtarr = _jax_get_dt_array(t_table)

    @jjit
    def scan_func(carryover, el):
        tacc, dt = el
        dmgas_dt = carryover

        lgtacc = jnp.log10(tacc)
        res = _calc_halo_history_scalar(lgtacc, *all_mah_params)
        dmhdt_at_tacc, log_mah_at_tacc = res
        dmgdt_inst = fb * dmhdt_at_tacc

        lag_factor = _gas_conversion_kern(t_form, tacc, dt, tau_dep, tau_dep_max)
        dmgas_dt_from_tacc = dmgdt_inst * lag_factor * dt
        dmgas_dt = dmgas_dt + dmgas_dt_from_tacc

        carryover = dmgas_dt
        accumulated = dmgas_dt
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = jnp.array((t_table, dtarr)).T
    res = lax.scan(scan_func, scan_init, scan_arr)
    dmgas_dt = res[0]
    sfr = dmgas_dt * sfr_eff
    return sfr


@jjit
def _sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi):
    """Instantaneous baryon conversion efficiency of main sequence galaxies

    Main sequence efficiency kernel, epsilon(Mhalo)

    Parameters
    ----------
    lgm : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun

    lgmcrit : float
        Base-10 log of the critical mass
    lgy_at_mcrit : float
        Base-10 log of the critical efficiency at critical mass

    indx_lo : float
        Asymptotic value of the efficiency at low halo masses

    indx_hi : float
        Asymptotic value of the efficiency at high halo masses

    Returns
    -------
    efficiency : ndarray of shape (n_times)
        Main sequence efficiency value at each snapshot

    """
    slope = _sigmoid(lgm, lgmcrit, INDX_K, indx_lo, indx_hi)
    eff = lgy_at_mcrit + slope * (lgm - lgmcrit)
    return 10**eff


@jjit
def _get_bounded_sfr_params(
    u_lgmcrit,
    u_lgy_at_mcrit,
    u_indx_lo,
    u_indx_hi,
    u_tau_dep,
):
    lgmcrit = _sigmoid(u_lgmcrit, *MS_BOUNDING_SIGMOID_PDICT["lgmcrit"])
    lgy_at_mcrit = _sigmoid(u_lgy_at_mcrit, *MS_BOUNDING_SIGMOID_PDICT["lgy_at_mcrit"])
    indx_lo = _sigmoid(u_indx_lo, *MS_BOUNDING_SIGMOID_PDICT["indx_lo"])
    indx_hi = _sigmoid(u_indx_hi, *MS_BOUNDING_SIGMOID_PDICT["indx_hi"])
    tau_dep = _sigmoid(u_tau_dep, *MS_BOUNDING_SIGMOID_PDICT["tau_dep"])
    bounded_params = (
        lgmcrit,
        lgy_at_mcrit,
        indx_lo,
        indx_hi,
        tau_dep,
    )
    return bounded_params


@jjit
def _get_unbounded_sfr_params(
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,
    tau_dep,
):
    u_lgmcrit = _inverse_sigmoid(lgmcrit, *MS_BOUNDING_SIGMOID_PDICT["lgmcrit"])
    u_lgy_at_mcrit = _inverse_sigmoid(
        lgy_at_mcrit, *MS_BOUNDING_SIGMOID_PDICT["lgy_at_mcrit"]
    )
    u_indx_lo = _inverse_sigmoid(indx_lo, *MS_BOUNDING_SIGMOID_PDICT["indx_lo"])
    u_indx_hi = _inverse_sigmoid(indx_hi, *MS_BOUNDING_SIGMOID_PDICT["indx_hi"])
    u_tau_dep = _inverse_sigmoid(tau_dep, *MS_BOUNDING_SIGMOID_PDICT["tau_dep"])
    bounded_params = (
        u_lgmcrit,
        u_lgy_at_mcrit,
        u_indx_lo,
        u_indx_hi,
        u_tau_dep,
    )
    return bounded_params


@jjit
def _get_bounded_sfr_params_galpop_kern(ms_params):
    return jnp.array(_get_bounded_sfr_params(*ms_params))


@jjit
def _get_unbounded_sfr_params_galpop_kern(u_ms_params):
    return jnp.array(_get_unbounded_sfr_params(*u_ms_params))


_get_bounded_sfr_params_vmap = jjit(
    vmap(_get_bounded_sfr_params_galpop_kern, in_axes=(0,))
)
_get_unbounded_sfr_params_vmap = jjit(
    vmap(_get_unbounded_sfr_params_galpop_kern, in_axes=(0,))
)


MSUParams = namedtuple("MSUParams", ["u_" + key for key in DEFAULT_MS_PDICT.keys()])
DEFAULT_U_MS_PARAMS = MSUParams(*_get_unbounded_sfr_params(*DEFAULT_MS_PARAMS))
DEFAULT_U_MS_PDICT = OrderedDict(
    [(key, val) for key, val in zip(DEFAULT_U_MS_PARAMS._fields, DEFAULT_U_MS_PARAMS)]
)
