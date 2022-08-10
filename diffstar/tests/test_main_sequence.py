"""
"""
import numpy as np
from jax import vmap
from jax import jit as jjit
from ..kernels.main_sequence import get_lax_ms_sfh_from_mah_kern
from .test_diffstar_is_frozen import calc_sfh_on_default_params, _get_default_mah_params


def test_get_main_sequence_kernel(n_t=400, n_steps=100):

    args, sfh = calc_sfh_on_default_params(n_t=n_t)
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args
    tarr = 10**lgt

    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index

    ms_sfh_from_mah_kern = get_lax_ms_sfh_from_mah_kern(lgt0=lgt0, n_steps=n_steps)
    ms_sfh_from_mah_vmap = jjit(vmap(ms_sfh_from_mah_kern, in_axes=[0, None, None]))
    lax_sfh = ms_sfh_from_mah_vmap(tarr, mah_params, u_ms_params)

    return tarr, sfh, lax_sfh

    assert np.allclose(sfh, lax_sfh, atol=1)
