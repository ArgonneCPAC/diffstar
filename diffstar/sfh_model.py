"""
"""
from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from .kernels.history_kernel_builders import build_sfh_from_mah_kernel
from .utils import cumulative_mstar_formed

_sfh_singlegal_kern = build_sfh_from_mah_kernel(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop="scan",
)

_sfh_galpop_kern = build_sfh_from_mah_kernel(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop="scan",
    galpop_loop="vmap",
)

_cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))

GalHistory = namedtuple("GalHistory", ("sfh", "smh"))


@jjit
def calc_sfh_smh_singlegal(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    args = (tarr, mah_params, sfh_params.ms_params, sfh_params.q_params, lgt0, fb)
    sfh = _sfh_singlegal_kern(*args)
    smh = cumulative_mstar_formed(tarr, sfh)
    return GalHistory(sfh, smh)


@jjit
def calc_sfh_smh_galpop(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    mah_params = jnp.array(mah_params).T
    ms_params = jnp.array(sfh_params.ms_params).T
    q_params = jnp.array(sfh_params.q_params).T
    args = tarr, mah_params, ms_params, q_params, lgt0, fb
    sfh = _sfh_galpop_kern(*args)
    smh = _cumulative_mstar_formed_vmap(tarr, sfh)
    return GalHistory(sfh, smh)
