import numpy as np
import jax.numpy as jnp

from diffstar.kernels.main_sequence_kernels import (
    _lax_ms_sfh_scalar_kern_scan,
    _lax_ms_sfh_scalar_kern_sum,
)
from diffstar.kernels.main_sequence_kernels import DEFAULT_MS_PARAMS
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar.defaults import T_TABLE_MIN, TODAY
from diffstar.defaults import FB


def test_main_sequence_kernels_lax_ms_sfh_scalar_kern_scan_vs_sum():
    lgt0 = jnp.log10(TODAY)
    t_form = 12.0
    t_table = jnp.linspace(T_TABLE_MIN, t_form, 20)

    np.testing.assert_allclose(
        _lax_ms_sfh_scalar_kern_scan(
            t_form, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, lgt0, FB, t_table
        ),
        _lax_ms_sfh_scalar_kern_sum(
            t_form, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, lgt0, FB, t_table
        ),
        rtol=1e-6,
        atol=1e-6,
    )
