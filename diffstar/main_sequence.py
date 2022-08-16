"""
"""
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap
from .constants import LGT0
from .kernels.gas_consumption import FB
from .kernels.main_sequence_kernels import DEFAULT_N_STEPS, DEFAULT_T_MIN
from .kernels.main_sequence_kernels import _lax_ms_sfh_scalar_kern
from .kernels.main_sequence_kernels import _get_bounded_sfr_params


def get_lax_ms_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    lgt0=LGT0,
    tacc_integration_min=DEFAULT_T_MIN,
    fb=FB,
    tobs_loop=None,
    galpop_loop=None,
):
    uniform_table = jnp.linspace(0, 1, n_steps)

    @jjit
    def _kern(t_form, mah_params, u_ms_params):
        ms_params = _get_bounded_sfr_params(*u_ms_params)
        tau_dep = ms_params[4]
        t_min = jnp.max(jnp.array((tacc_integration_min, t_form - tau_dep)))
        t_table = t_min + uniform_table * (t_form - t_min)
        args = t_form, mah_params, ms_params, lgt0, fb, t_table
        return _lax_ms_sfh_scalar_kern(*args)

    kern_with_tobs_loop = _get_kern_with_tobs_loop(_kern, tobs_loop)
    lax_ms_sfh_from_mah_kern = _get_kern_with_galpop_loop(
        kern_with_tobs_loop, galpop_loop
    )

    return lax_ms_sfh_from_mah_kern


def _get_kern_with_tobs_loop(kern, tobs_loop):
    if tobs_loop == "vmap":
        _t = [0, None, None]
        new_kern = jjit(vmap(kern, in_axes=_t))
    elif tobs_loop == "scan":

        @jjit
        def new_kern(tarr, mah_params, u_ms_params):
            @jjit
            def scan_func_time_array(carryover, el):
                t_form = el
                sfr_at_t_form = kern(t_form, mah_params, u_ms_params)
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
        _g = [None, 0, 0]
        new_kern = jjit(vmap(kern, in_axes=_g))
    elif galpop_loop == "scan":

        @jjit
        def new_kern(t, mah_params_galpop, ms_u_params_galpop):
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
                sfh_galpop = kern(t, mah_params, u_ms_params)
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