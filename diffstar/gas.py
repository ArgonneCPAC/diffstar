"""Calculate mass of gas available for star formation from the freshly accreted gas.
"""
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap
from .utils import tw_bin_jax_kern

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


_a, _b = (0, None, 0, None, None), (None, 0, None, None, None)
_depletion_kernel = jjit(vmap(vmap(_gas_conversion_kern, in_axes=_b), in_axes=_a))


@jjit
def _get_lagged_gas(lgt, dt, dmhdt, tau_dep, tau_dep_max):
    t_table = 10**lgt
    mgas_inst = FB * dmhdt

    depletion_matrix = _depletion_kernel(t_table, t_table, dt, tau_dep, tau_dep_max)
    depletion_matrix_inst = jnp.identity(len(lgt)) / dt
    tau_w = jnp.where(
        tau_dep > 5.0 * jnp.mean(dt), jnp.ones(len(dt)), jnp.zeros(len(dt))
    )
    depletion_matrix = jnp.where(tau_w == 1, depletion_matrix, depletion_matrix_inst)

    integrand = mgas_inst * depletion_matrix * dt
    lagged_mgas = jnp.sum(integrand, axis=1)

    return lagged_mgas


@jjit
def _dmgas_dt_kern(t, t_table, dt, dmhdt_table, tau_dep, tau_dep_max):
    dmgas_dt_table = FB * dmhdt_table

    @jjit
    def scan_mgas_at_t(carryover, el):
        dmgas_dt = carryover
        tacc, dmgdt_inst = el

        lag_factor = _gas_conversion_kern(t, tacc, dt, tau_dep, tau_dep_max)
        dmgas_dt_from_tacc = dmgdt_inst * lag_factor * dt
        dmgas_dt = dmgas_dt + dmgas_dt_from_tacc

        carryover = dmgas_dt
        accumulated = dmgas_dt
        return carryover, accumulated

    dmgas_dt_init = 0.0
    arr = jnp.vstack((t_table, dmgas_dt_table)).T
    res = lax.scan(scan_mgas_at_t, dmgas_dt_init, arr)
    dmgas_dt = res[0]

    return dmgas_dt
