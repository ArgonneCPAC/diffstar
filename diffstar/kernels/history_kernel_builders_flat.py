""" """

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

from ..defaults_flat import SFR_MIN
from .main_sequence_kernels_flat import sfh_ms_kernel
from .quenching_kernels_flat import _quenching_kern


@jjit
def _sfh_singlegal_scalar(tform, mah_params, ms_params, q_params, logt0, fb):
    lgt_form = jnp.log10(tform)

    sfr_ms = sfh_ms_kernel(tform, mah_params, ms_params, logt0, fb)

    lg_qt, qlglgdt, lg_drop, lg_rejuv = q_params
    lg_q_dt = 10**qlglgdt
    q_params = (lg_qt, lg_q_dt, lg_drop, lg_rejuv)
    qfunc = _quenching_kern(lgt_form, *q_params)

    sfr = qfunc * sfr_ms
    sfr = lax.cond(sfr < SFR_MIN, lambda x: SFR_MIN, lambda x: x, sfr)

    return sfr


_in = [0, None, None, None, None, None]
_sfh_singlegal_kern = jjit(vmap(_sfh_singlegal_scalar, in_axes=_in))


def _sfh_galpop_kern_kern(tform, mah_params, ms_params, q_params, logt0, fb):
    _mah_params = DEFAULT_MAH_PARAMS._make(mah_params)
    return _sfh_singlegal_kern(tform, _mah_params, ms_params, q_params, logt0, fb)


_in = [None, 0, 0, 0, None, None]
_sfh_galpop_kern = jjit(vmap(_sfh_galpop_kern_kern, in_axes=_in))
