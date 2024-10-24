"""
"""

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from ..defaults import DEFAULT_DIFFSTAR_PARAMS, DEFAULT_N_STEPS, SFR_MIN, T_BIRTH_MIN
from .main_sequence_kernels_tpeak import (
    _lax_ms_sfh_scalar_kern_scan,
    _lax_ms_sfh_scalar_kern_sum,
)
from .quenching_kernels import _quenching_kern

__all__ = ("build_sfh_from_mah_kernel",)

N_MAH_PARAMS = len(DEFAULT_MAH_PARAMS)
N_MS_PARAMS = len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
N_Q_PARAMS = len(DEFAULT_DIFFSTAR_PARAMS.q_params)


def build_sfh_from_mah_kernel(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop=None,
    galpop_loop=None,
    tform_loop="sum",
):
    """Build a JAX-jitted kernel to calculate SFHs of a galaxy population.

    Parameters
    ----------
    n_steps : int, optional
        Number of timesteps to use in the tacc integration

    tacc_integration_min : float, optional
        Earliest time to use in the tacc integrations. Default is 0.01 Gyr.

    tobs_loop : string, optional
        Argument specifies whether the input time of observation is a scalar or array
        Default argument is None, for a JAX kernel that assumes scalar input for tobs
        For a JAX kernel that assumes an array input for tobs,
        options are either 'vmap' or 'scan', specifying the calculation method

    galpop_loop : string, optional
        Argument specifies whether the input galaxy/halo parameters assumed by the
        returned JAX kernel pertain to a single galaxy or a population.
        Default argument is None, for a single-galaxy JAX kernel
        For a JAX kernel that assumes galaxy population,
        options are either 'vmap' or 'scan', specifying the calculation method

    tform_loop : string
        Use 'sum' for faster vmap-based calculation and 'scan' for slower alternative.
        Default is 'sum'

    Returns
    -------
    sfh_from_mah_kern : function
        JAX-jitted function that calculates SFH in accord with the input arguments
        Function signature is as follows:

        def sfh_from_mah_kern(t, mah_params, ms_params, q_params, lgt0, fb):
            return sfh

    """
    if tform_loop == "sum":
        _lax_ms_sfh_scalar_kern = _lax_ms_sfh_scalar_kern_sum
    elif tform_loop == "scan":
        _lax_ms_sfh_scalar_kern = _lax_ms_sfh_scalar_kern_scan

    uniform_table = jnp.linspace(0, 1, n_steps)

    @jjit
    def _kern(
        t_form,
        logmp,
        logtc,
        early_index,
        late_index,
        t_peak,
        lgmcrit,
        lgy_at_mcrit,
        indx_lo,
        indx_hi,
        tau_dep,
        lg_qt,
        qlglgdt,
        lg_drop,
        lg_rejuv,
        lgt0,
        fb,
    ):
        mah_params = logmp, logtc, early_index, late_index, t_peak
        mah_params = DEFAULT_MAH_PARAMS._make(mah_params)
        ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep

        t_min = jnp.max(jnp.array((tacc_integration_min, t_form - tau_dep)))
        t_table = t_min + uniform_table * (t_form - t_min)
        args = t_form, mah_params, ms_params, lgt0, fb, t_table
        ms_sfr = _lax_ms_sfh_scalar_kern(*args)
        lgt_form = jnp.log10(t_form)

        lg_q_dt = 10**qlglgdt
        qkern_inputs = lg_qt, lg_q_dt, lg_drop, lg_rejuv
        qfunc = _quenching_kern(lgt_form, *qkern_inputs)
        sfr = qfunc * ms_sfr
        sfr = lax.cond(sfr < SFR_MIN, lambda x: SFR_MIN, lambda x: x, sfr)
        return sfr

    kern_with_tobs_loop = _get_kern_with_tobs_loop(_kern, tobs_loop)
    sfh_from_mah_kern = _get_kern_with_galpop_loop(kern_with_tobs_loop, galpop_loop)

    return sfh_from_mah_kern


def _get_kern_with_tobs_loop(kern, tobs_loop):
    if tobs_loop == "vmap":
        _t = [
            0,
            *[None] * N_MAH_PARAMS,
            *[None] * N_MS_PARAMS,
            *[None] * N_Q_PARAMS,
            None,
            None,
        ]
        new_kern = jjit(vmap(kern, in_axes=_t))
    elif tobs_loop == "scan":

        @jjit
        def new_kern(
            tarr,
            logmp,
            logtc,
            early_index,
            late_index,
            t_peak,
            lgmcrit,
            lgy_at_mcrit,
            indx_lo,
            indx_hi,
            tau_dep,
            lg_qt,
            qlglgdt,
            lg_drop,
            lg_rejuv,
            lgt0,
            fb,
        ):
            @jjit
            def scan_func_time_array(carryover, el):
                t_form = el
                sfr_at_t_form = kern(
                    t_form,
                    logmp,
                    logtc,
                    early_index,
                    late_index,
                    t_peak,
                    lgmcrit,
                    lgy_at_mcrit,
                    indx_lo,
                    indx_hi,
                    tau_dep,
                    lg_qt,
                    qlglgdt,
                    lg_drop,
                    lg_rejuv,
                    lgt0,
                    fb,
                )
                carryover = sfr_at_t_form
                accumulated = sfr_at_t_form
                return carryover, accumulated

            scan_init = 0.0
            scan_arr = tarr
            res = lax.scan(scan_func_time_array, scan_init, scan_arr)
            sfh = res[1]
            return sfh

    elif tobs_loop is None:
        new_kern = kern
    else:
        msg = "Input `tobs_loop`={0} must be either `vmap` or `scan`"
        raise ValueError(msg.format(tobs_loop))
    return new_kern


def _get_kern_with_galpop_loop(kern, galpop_loop):
    if galpop_loop == "vmap":
        _g = [
            None,
            *[0] * N_MAH_PARAMS,
            *[0] * N_MS_PARAMS,
            *[0] * N_Q_PARAMS,
            None,
            None,
        ]
        new_kern = jjit(vmap(kern, in_axes=_g))
    elif galpop_loop == "scan":

        @jjit
        def new_kern(
            t,
            logmp,
            logtc,
            early_index,
            late_index,
            t_peak,
            lgmcrit,
            lgy_at_mcrit,
            indx_lo,
            indx_hi,
            tau_dep,
            lg_qt,
            qlglgdt,
            lg_drop,
            lg_rejuv,
            lgt0,
            fb,
        ):
            n_gals = logmp.shape[0]

            n_params = N_MAH_PARAMS + N_MS_PARAMS + N_Q_PARAMS
            galpop_params = jnp.zeros(shape=(n_gals, n_params))
            galpop_params = galpop_params.at[:, 0].set(logmp)
            galpop_params = galpop_params.at[:, 1].set(logtc)
            galpop_params = galpop_params.at[:, 2].set(early_index)
            galpop_params = galpop_params.at[:, 3].set(late_index)
            galpop_params = galpop_params.at[:, 4].set(t_peak)

            galpop_params = galpop_params.at[:, 5].set(lgmcrit)
            galpop_params = galpop_params.at[:, 6].set(lgy_at_mcrit)
            galpop_params = galpop_params.at[:, 7].set(indx_lo)
            galpop_params = galpop_params.at[:, 8].set(indx_hi)
            galpop_params = galpop_params.at[:, 9].set(tau_dep)

            galpop_params = galpop_params.at[:, 10].set(lg_qt)
            galpop_params = galpop_params.at[:, 11].set(qlglgdt)
            galpop_params = galpop_params.at[:, 12].set(lg_drop)
            galpop_params = galpop_params.at[:, 13].set(lg_rejuv)

            @jjit
            def scan_func_galpop(carryover, el):
                params = el
                mah_params = params[:N_MAH_PARAMS]
                i, j = N_MAH_PARAMS, N_MAH_PARAMS + N_MS_PARAMS
                ms_params = params[i:j]
                i = N_MAH_PARAMS + N_MS_PARAMS
                q_params = params[i:]
                sfh_galpop = kern(t, *mah_params, *ms_params, *q_params, lgt0, fb)
                carryover = sfh_galpop
                accumulated = sfh_galpop
                return carryover, accumulated

            scan_init = jnp.zeros_like(t)
            scan_arr = galpop_params
            res = lax.scan(scan_func_galpop, scan_init, scan_arr)
            sfh_galpop = res[1]
            return sfh_galpop

    elif galpop_loop is None:
        new_kern = kern
    else:
        msg = "Input `galpop_loop`={0} must be either `vmap` or `scan`"
        raise ValueError(msg.format(galpop_loop))
    return new_kern
