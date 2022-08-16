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


def get_lax_ms_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    lgt0=LGT0,
    t_min=DEFAULT_T_MIN,
    fb=FB,
    time_array=None,
    galpop=None,
):
    @jjit
    def _lax_ms_sfh_from_mah_kern(t_form, mah_params, u_ms_params):
        args = t_form, mah_params, u_ms_params, n_steps, lgt0, t_min, fb
        return _lax_ms_sfh_scalar_kern(*args)

    if time_array == "vmap":
        _t = [0, None, None]
        ret_func0 = jjit(vmap(_lax_ms_sfh_from_mah_kern, in_axes=_t))
    elif time_array == "scan":

        @jjit
        def ret_func0(tarr, mah_params, u_ms_params):
            @jjit
            def scan_func_time_array(carryover, el):
                t_form = el
                sfr_at_t_form = _lax_ms_sfh_from_mah_kern(
                    t_form, mah_params, u_ms_params
                )
                carryover = sfr_at_t_form
                accumulated = sfr_at_t_form
                return carryover, accumulated

            scan_init = 0.0
            scan_arr = tarr
            res = lax.scan(scan_func_time_array, scan_init, scan_arr)
            sfh = res[1]
            return sfh

    elif time_array is None:
        ret_func0 = _lax_ms_sfh_from_mah_kern
    else:
        msg = "Input `time_array`={0} must be either `vmap` or `scan`"
        raise ValueError(msg.format(time_array))

    if galpop == "vmap":
        _g = [None, 0, 0]
        ret_func = jjit(vmap(ret_func0, in_axes=_g))
    elif galpop == "scan":

        @jjit
        def ret_func(t, mah_params_galpop, ms_u_params_galpop):
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
                sfh_galpop = ret_func0(t, mah_params, u_ms_params)
                carryover = sfh_galpop
                accumulated = sfh_galpop
                return carryover, accumulated

            scan_init = jnp.zeros_like(t)
            scan_arr = galpop_params
            res = lax.scan(scan_func_galpop, scan_init, scan_arr)
            sfh_galpop = res[1]
            return sfh_galpop

    elif galpop is None:
        ret_func = ret_func0
    else:
        msg = "Input `galpop`={0} must be either `vmap` or `scan`"
        raise ValueError(msg.format(galpop))

    return ret_func
