"""
"""
from jax import jit as jjit
from jax import vmap

from .defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from .kernels.history_kernel_builders import build_sfh_from_mah_kernel
from .utils import cumulative_mstar_formed

_sfh_singlegal_kern = build_sfh_from_mah_kernel(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop="scan",
)

_cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))


@jjit
def calc_sfh_smh_singlegal(sfh_params, mah_params, tarr, lgt0=LGT0, fb=FB):
    args = (tarr, mah_params, sfh_params.ms_params, sfh_params.q_params, lgt0, fb)
    sfh = _sfh_singlegal_kern(*args)
    smh = cumulative_mstar_formed(tarr, sfh)
    return sfh, smh