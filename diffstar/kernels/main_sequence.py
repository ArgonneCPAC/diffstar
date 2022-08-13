"""
"""
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import grad
from jax import vmap
from diffmah.individual_halo_assembly import _rolling_plaw_vs_t, _rolling_plaw_vs_logt
from diffmah.individual_halo_assembly import DEFAULT_MAH_PARAMS
from ..utils import _jax_get_dt_array
from ..stars import _get_bounded_sfr_params, _sfr_eff_plaw, SFR_PARAM_BOUNDS
from ..stars import LGT0
from .gas_consumption import _gas_conversion_kern, FB

_d_log_mh_dt_scalar = jjit(grad(_rolling_plaw_vs_t, argnums=0))

DEFAULT_N_STEPS = 50
DEFAULT_T_MIN = 0.01


@jjit
def _dmhalo_dt_scalar(t, log_mah, lgt0, logmp, logtc, mah_k, early, late):
    d_log_mh_dt = _d_log_mh_dt_scalar(t, lgt0, logmp, logtc, mah_k, early, late)
    dmhdt = d_log_mh_dt * (10.0 ** (log_mah - 9.0)) / jnp.log10(jnp.e)
    return dmhdt


def _lax_ms_sfh_from_mah_closure_input(
    t_form, mah_params, u_ms_params, n_steps, lgt0, t_min, fb
):

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


def get_lax_ms_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS, lgt0=LGT0, t_min=DEFAULT_T_MIN, fb=FB, vmap_time=True
):
    @jjit
    def _lax_ms_sfh_from_mah_kern(t_form, mah_params, u_ms_params):
        args = t_form, mah_params, u_ms_params, n_steps, lgt0, t_min, fb
        return _lax_ms_sfh_from_mah_closure_input(*args)

    if vmap_time:
        _t = [0, None, None]
        _lax_ms_sfh_from_mah = jjit(vmap(_lax_ms_sfh_from_mah_kern, in_axes=_t))
    else:
        _lax_ms_sfh_from_mah = _lax_ms_sfh_from_mah_kern

    return _lax_ms_sfh_from_mah


def scan_sfhpop(sfh_kern_at_tform, tarr, mah_params, u_ms_params):
    pass

    @jjit
    def scan_func(carryover, el):
        t_form = el
        sfr_at_t_form = sfh_kern_at_tform(t_form, mah_params, u_ms_params)
        carryover = sfr_at_t_form
        accumulated = sfr_at_t_form
        return carryover, accumulated

    scan_init = 0.0
    scan_arr = tarr
    sfh = lax.scan(scan_func, scan_init, scan_arr)
    return sfh
