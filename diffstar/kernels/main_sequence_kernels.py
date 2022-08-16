"""
"""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import grad
from jax import numpy as jnp
from jax import lax
from jax import vmap
from ..utils import _sigmoid, _inverse_sigmoid, _jax_get_dt_array
from diffmah.individual_halo_assembly import DEFAULT_MAH_PARAMS
from diffmah.individual_halo_assembly import _rolling_plaw_vs_t, _rolling_plaw_vs_logt
from .gas_consumption import _gas_conversion_kern, _get_lagged_gas


_d_log_mh_dt_scalar = jjit(grad(_rolling_plaw_vs_t, argnums=0))

INDX_K = 9.0  # Main sequence efficiency transition speed.

DEFAULT_N_STEPS = 50
DEFAULT_T_MIN = 0.01

_SFR_PARAM_BOUNDS = OrderedDict(
    lgmcrit=(9.0, 13.5),
    lgy_at_mcrit=(-3.0, 0.0),
    indx_lo=(0.0, 5.0),
    indx_hi=(-5.0, 0.0),
    tau_dep=(0.0, 20.0),
)


def calculate_sigmoid_bounds(param_bounds):
    bounds_out = OrderedDict()

    for key in param_bounds:
        _bounds = (
            float(np.mean(param_bounds[key])),
            abs(float(4.0 / np.diff(param_bounds[key]))),
        )
        bounds_out[key] = _bounds + param_bounds[key]
    return bounds_out


SFR_PARAM_BOUNDS = calculate_sigmoid_bounds(_SFR_PARAM_BOUNDS)


@jjit
def _dmhalo_dt_scalar(t, log_mah, lgt0, logmp, logtc, mah_k, early, late):
    d_log_mh_dt = _d_log_mh_dt_scalar(t, lgt0, logmp, logtc, mah_k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt


def _lax_ms_sfh_scalar_kern(t_form, mah_params, u_ms_params, n_steps, lgt0, t_min, fb):

    mah_k = DEFAULT_MAH_PARAMS["mah_k"]
    logmp, logtc, early, late = mah_params
    all_mah_params = lgt0, logmp, logtc, mah_k, early, late
    lgt_form = jnp.log10(t_form)
    log_mah_at_tform = _rolling_plaw_vs_logt(lgt_form, *all_mah_params)

    ms_params = _get_bounded_sfr_params(*u_ms_params)
    sfr_eff_params = ms_params[:4]
    sfr_eff = _sfr_eff_plaw(log_mah_at_tform, *sfr_eff_params)

    tau_dep = ms_params[4]
    tau_dep_max = SFR_PARAM_BOUNDS["tau_dep"][3]

    t_min = jnp.max(jnp.array((t_min, t_form - tau_dep)))
    t_table = jnp.linspace(t_min, t_form, n_steps)
    dtarr = _jax_get_dt_array(t_table)

    @jjit
    def scan_func(carryover, el):
        tacc, dt = el
        dmgas_dt = carryover

        log_mah_at_tacc = _rolling_plaw_vs_logt(jnp.log10(tacc), *all_mah_params)
        dmhdt = _dmhalo_dt_scalar(
            tacc, log_mah_at_tacc, lgt0, logmp, logtc, mah_k, early, late
        )
        dmgdt_inst = fb * dmhdt

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
    lgmcrit = _sigmoid(u_lgmcrit, *SFR_PARAM_BOUNDS["lgmcrit"])
    lgy_at_mcrit = _sigmoid(u_lgy_at_mcrit, *SFR_PARAM_BOUNDS["lgy_at_mcrit"])
    indx_lo = _sigmoid(u_indx_lo, *SFR_PARAM_BOUNDS["indx_lo"])
    indx_hi = _sigmoid(u_indx_hi, *SFR_PARAM_BOUNDS["indx_hi"])
    tau_dep = _sigmoid(u_tau_dep, *SFR_PARAM_BOUNDS["tau_dep"])
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
    u_lgmcrit = _inverse_sigmoid(lgmcrit, *SFR_PARAM_BOUNDS["lgmcrit"])
    u_lgy_at_mcrit = _inverse_sigmoid(lgy_at_mcrit, *SFR_PARAM_BOUNDS["lgy_at_mcrit"])
    u_indx_lo = _inverse_sigmoid(indx_lo, *SFR_PARAM_BOUNDS["indx_lo"])
    u_indx_hi = _inverse_sigmoid(indx_hi, *SFR_PARAM_BOUNDS["indx_hi"])
    u_tau_dep = _inverse_sigmoid(tau_dep, *SFR_PARAM_BOUNDS["tau_dep"])
    bounded_params = (
        u_lgmcrit,
        u_lgy_at_mcrit,
        u_indx_lo,
        u_indx_hi,
        u_tau_dep,
    )
    return bounded_params


_get_bounded_sfr_params_vmap = jjit(vmap(_get_bounded_sfr_params, (0,) * 5, 0))
_get_unbounded_sfr_params_vmap = jjit(vmap(_get_unbounded_sfr_params, (0,) * 5, 0))


@jjit
def _ms_sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params):
    """Main Sequence formation history of an individual galaxy."""

    bounded_params = _get_bounded_sfr_params(*sfr_params)
    sfr_ms_params = bounded_params[:4]
    tau_dep = bounded_params[4]
    efficiency = _sfr_eff_plaw(log_mah, *sfr_ms_params)

    tau_dep_max = SFR_PARAM_BOUNDS["tau_dep"][3]
    lagged_mgas = _get_lagged_gas(lgt, dtarr, dmhdt, tau_dep, tau_dep_max)

    ms_sfr = lagged_mgas * efficiency
    return ms_sfr
