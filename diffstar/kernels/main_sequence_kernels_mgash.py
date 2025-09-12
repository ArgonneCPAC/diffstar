""" """

from collections import OrderedDict, namedtuple

import numpy as np
from diffmah.diffmah_kernels import _diffmah_kern, _diffmah_kern_scalar, _log_mah_kern
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _jax_get_dt_array, _sigmoid

DEFAULT_MS_PDICT = OrderedDict(
    lgmcrit=12.0,
    lgy_at_mcrit=-10.0,
    indx_lo=1.0,
    indx_hi=-1.0,
)
MSParams = namedtuple("MSParams", list(DEFAULT_MS_PDICT.keys()))
DEFAULT_MS_PARAMS = MSParams(*list(DEFAULT_MS_PDICT.values()))

MS_PARAM_BOUNDS_PDICT = OrderedDict(
    lgmcrit=(9.0, 13.5),
    lgy_at_mcrit=(-12.0, -8.0),
    indx_lo=(0.0, 5.0),
    indx_hi=(-5.0, 0.0),
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
def _sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi):
    slope = _sigmoid(lgm, lgmcrit, INDX_K, indx_lo, indx_hi)
    eff = lgy_at_mcrit + slope * (lgm - lgmcrit)
    return eff


@jjit
def sfh_ms_kernel(tform, mah_params, ms_params, logt0, fb):
    log_mah = _log_mah_kern(mah_params, tform, logt0)
    Mgash = fb * 10**log_mah
    sfh_ms = Mgash * 10 ** _sfr_eff_plaw(log_mah, *ms_params)
    return sfh_ms


_in = [0, None, None, None, None]
sfh_ms_kernel_vmap = jjit(vmap(sfh_ms_kernel, in_axes=_in))


@jjit
def _get_bounded_sfr_params(
    u_lgmcrit,
    u_lgy_at_mcrit,
    u_indx_lo,
    u_indx_hi,
):
    lgmcrit = _sigmoid(u_lgmcrit, *MS_BOUNDING_SIGMOID_PDICT["lgmcrit"])
    lgy_at_mcrit = _sigmoid(u_lgy_at_mcrit, *MS_BOUNDING_SIGMOID_PDICT["lgy_at_mcrit"])
    indx_lo = _sigmoid(u_indx_lo, *MS_BOUNDING_SIGMOID_PDICT["indx_lo"])
    indx_hi = _sigmoid(u_indx_hi, *MS_BOUNDING_SIGMOID_PDICT["indx_hi"])
    bounded_params = (
        lgmcrit,
        lgy_at_mcrit,
        indx_lo,
        indx_hi,
    )
    return bounded_params


@jjit
def _get_unbounded_sfr_params(
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,
):
    u_lgmcrit = _inverse_sigmoid(lgmcrit, *MS_BOUNDING_SIGMOID_PDICT["lgmcrit"])
    u_lgy_at_mcrit = _inverse_sigmoid(
        lgy_at_mcrit, *MS_BOUNDING_SIGMOID_PDICT["lgy_at_mcrit"]
    )
    u_indx_lo = _inverse_sigmoid(indx_lo, *MS_BOUNDING_SIGMOID_PDICT["indx_lo"])
    u_indx_hi = _inverse_sigmoid(indx_hi, *MS_BOUNDING_SIGMOID_PDICT["indx_hi"])
    bounded_params = (
        u_lgmcrit,
        u_lgy_at_mcrit,
        u_indx_lo,
        u_indx_hi,
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
