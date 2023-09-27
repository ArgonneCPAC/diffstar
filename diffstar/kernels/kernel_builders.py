"""
"""
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from ..defaults import DEFAULT_N_STEPS, SFR_MIN, T_BIRTH_MIN
from .main_sequence_kernels import _get_bounded_sfr_params, _lax_ms_sfh_scalar_kern
from .quenching_kernels import _quenching_kern_u_params

__all__ = ("get_sfh_from_mah_kern", "get_ms_sfh_from_mah_kern")


def get_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop=None,
    galpop_loop=None,
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

    Returns
    -------
    sfh_from_mah_kern : function
        JAX-jitted function that calculates SFH in accord with the input arguments
        Function signature is as follows:

        def sfh_from_mah_kern(t, mah_params, u_ms_params, u_q_params, lgt0, fb):
            return sfh

    """
    uniform_table = jnp.linspace(0, 1, n_steps)

    @jjit
    def _kern(t_form, mah_params, u_ms_params, u_q_params, lgt0, fb):
        ms_params = _get_bounded_sfr_params(*u_ms_params)
        tau_dep = ms_params[4]
        t_min = jnp.max(jnp.array((tacc_integration_min, t_form - tau_dep)))
        t_table = t_min + uniform_table * (t_form - t_min)
        args = t_form, mah_params, ms_params, lgt0, fb, t_table
        ms_sfr = _lax_ms_sfh_scalar_kern(*args)
        lgt_form = jnp.log10(t_form)
        qfunc = _quenching_kern_u_params(lgt_form, *u_q_params)
        sfr = qfunc * ms_sfr
        sfr = lax.cond(sfr < SFR_MIN, lambda x: SFR_MIN, lambda x: x, sfr)
        return sfr

    kern_with_tobs_loop = _get_kern_with_tobs_loop(_kern, tobs_loop)
    sfh_from_mah_kern = _get_kern_with_galpop_loop(kern_with_tobs_loop, galpop_loop)

    return sfh_from_mah_kern


def _get_kern_with_tobs_loop(kern, tobs_loop):
    if tobs_loop == "vmap":
        _t = [0, None, None, None, None, None]
        new_kern = jjit(vmap(kern, in_axes=_t))
    elif tobs_loop == "scan":

        @jjit
        def new_kern(tarr, mah_params, u_ms_params, u_q_params, lgt0, fb):
            @jjit
            def scan_func_time_array(carryover, el):
                t_form = el
                sfr_at_t_form = kern(
                    t_form, mah_params, u_ms_params, u_q_params, lgt0, fb
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
        _g = [None, 0, 0, 0, None, None]
        new_kern = jjit(vmap(kern, in_axes=_g))
    elif galpop_loop == "scan":

        @jjit
        def new_kern(
            t, mah_params_galpop, ms_u_params_galpop, q_u_params_galpop, lgt0, fb
        ):
            n_gals, n_mah_params = mah_params_galpop.shape
            n_ms_params = ms_u_params_galpop.shape[1]
            n_q_params = q_u_params_galpop.shape[1]
            n_params = n_mah_params + n_ms_params + n_q_params
            galpop_params = jnp.zeros(shape=(n_gals, n_params))
            galpop_params = galpop_params.at[:, :n_mah_params].set(mah_params_galpop)
            i, j = n_mah_params, n_mah_params + n_ms_params
            galpop_params = galpop_params.at[:, i:j].set(ms_u_params_galpop)
            i = n_mah_params + n_ms_params
            galpop_params = galpop_params.at[:, i:].set(q_u_params_galpop)

            @jjit
            def scan_func_galpop(carryover, el):
                params = el
                mah_params = params[:n_mah_params]
                i, j = n_mah_params, n_mah_params + n_ms_params
                u_ms_params = params[i:j]
                i = n_mah_params + n_ms_params
                u_q_params = params[i:]
                sfh_galpop = kern(t, mah_params, u_ms_params, u_q_params, lgt0, fb)
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


def get_ms_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop=None,
    galpop_loop=None,
):
    uniform_table = jnp.linspace(0, 1, n_steps)

    @jjit
    def _kern(t_form, mah_params, u_ms_params, lgt0, fb):
        ms_params = _get_bounded_sfr_params(*u_ms_params)
        tau_dep = ms_params[4]
        t_min = jnp.max(jnp.array((tacc_integration_min, t_form - tau_dep)))
        t_table = t_min + uniform_table * (t_form - t_min)
        args = t_form, mah_params, ms_params, lgt0, fb, t_table
        sfr = _lax_ms_sfh_scalar_kern(*args)
        sfr = lax.cond(sfr < SFR_MIN, lambda x: SFR_MIN, lambda x: x, sfr)
        return sfr

    kern_with_tobs_loop = _get_ms_kern_with_tobs_loop(_kern, tobs_loop)
    lax_ms_sfh_from_mah_kern = _get_ms_kern_with_galpop_loop(
        kern_with_tobs_loop, galpop_loop
    )

    return lax_ms_sfh_from_mah_kern


def _get_ms_kern_with_tobs_loop(kern, tobs_loop):
    if tobs_loop == "vmap":
        _t = [0, None, None, None, None]
        new_kern = jjit(vmap(kern, in_axes=_t))
    elif tobs_loop == "scan":

        @jjit
        def new_kern(tarr, mah_params, u_ms_params, lgt0, fb):
            @jjit
            def scan_func_time_array(carryover, el):
                t_form = el
                sfr_at_t_form = kern(t_form, mah_params, u_ms_params, lgt0, fb)
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


def _get_ms_kern_with_galpop_loop(kern, galpop_loop):
    if galpop_loop == "vmap":
        _g = [None, 0, 0, None, None]
        new_kern = jjit(vmap(kern, in_axes=_g))
    elif galpop_loop == "scan":

        @jjit
        def new_kern(t, mah_params_galpop, ms_u_params_galpop, lgt0, fb):
            n_gals, n_mah_params = mah_params_galpop.shape
            n_ms_params = ms_u_params_galpop.shape[1]
            n_params = n_mah_params + n_ms_params
            galpop_params = jnp.zeros(shape=(n_gals, n_params))
            galpop_params = galpop_params.at[:, :n_mah_params].set(mah_params_galpop)
            galpop_params = galpop_params.at[:, n_mah_params:].set(ms_u_params_galpop)

            @jjit
            def scan_func_galpop(carryover, el):
                params = el
                mah_params = params[:n_mah_params]
                u_ms_params = params[n_mah_params:]
                sfh_galpop = kern(t, mah_params, u_ms_params, lgt0, fb)
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
