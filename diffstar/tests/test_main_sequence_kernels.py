import numpy as np
import jax.numpy as jnp

from diffstar.kernels.main_sequence_kernels import (
    _lax_ms_sfh_scalar_kern_scan,
    _lax_ms_sfh_scalar_kern_sum,
)
from diffstar.kernels.main_sequence_kernels import DEFAULT_MS_PARAMS
from diffmah.defaults import DEFAULT_MAH_PARAMS, LGT0
from diffstar.defaults import T_TABLE_MIN, TODAY
from diffstar.defaults import FB

from diffstar.kernels.main_sequence_kernels import (
    _lax_ms_sfh_scalar_kern_sum as kern_sum,
    _lax_ms_sfh_scalar_kern_scan as kern_scan,
)
from diffstar.kernels.main_sequence_kernels_tpeak import (
    _lax_ms_sfh_scalar_kern_sum as kern_sum_tpeak,
    _lax_ms_sfh_scalar_kern_scan as kern_scan_tpeak,
)


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

def test_main_sequence_kernels_tpeak():

    t_table = jnp.logspace(0, LGT0, 100)
    t_form = jnp.logspace(0, LGT0, 30)
    t_peak = 16.0

    sfr_sum = np.array([kern_sum(x, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table) for x in t_form])
    sfr_scan = np.array([kern_scan(x, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table) for x in t_form])
    sfr_sum_tpeak = np.array([kern_sum_tpeak(x, DEFAULT_MAH_PARAMS, t_peak, DEFAULT_MS_PARAMS, LGT0, FB, t_table) for x in t_form])
    sfr_scan_tpeak = np.array([kern_scan_tpeak(x, DEFAULT_MAH_PARAMS, t_peak, DEFAULT_MS_PARAMS, LGT0, FB, t_table) for x in t_form])

    assert np.allclose(sfr_sum, sfr_sum_tpeak)
    assert np.allclose(sfr_scan, sfr_scan_tpeak)