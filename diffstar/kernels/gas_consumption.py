"""Calculate mass of gas available for star formation from the freshly accreted gas.
"""
from jax import jit as jjit
from jax import lax, vmap
from jax import numpy as jnp

from ..utils import tw_bin_jax_kern

FB = 0.156


@jjit
def _gas_conversion_kern(t_form, t_acc, dt, tau_dep, tau_dep_max):
    alpha = (tau_dep / 2.0) * (tau_dep / tau_dep_max)
    w = (tau_dep - alpha) / 3.0
    m = t_acc + alpha

    _norm = tw_bin_jax_kern(m, w, t_acc, t_acc + tau_dep)
    _norm = 1.0 / jnp.clip(_norm, 0.01, jnp.inf)

    tri_kern = lax.cond(
        t_form < t_acc,
        lambda x: 0.0,
        lambda x: _norm * tw_bin_jax_kern(m, w, x, x + dt) / dt,
        t_form,
    )
    return tri_kern


_vmap_gas_conversion_kern = jjit(vmap(
    _gas_conversion_kern,
    in_axes=(None, 0, None, None, None)
))
