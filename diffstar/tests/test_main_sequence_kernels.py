import jax.numpy as jnp
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS as OLD_DEFAULT_MAH_PARAMS
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS

from ..defaults import FB, LGT0, T_TABLE_MIN, TODAY
from ..kernels.main_sequence_kernels import DEFAULT_MS_PARAMS
from ..kernels.main_sequence_kernels import _lax_ms_sfh_scalar_kern_scan
from ..kernels.main_sequence_kernels import _lax_ms_sfh_scalar_kern_scan as kern_scan
from ..kernels.main_sequence_kernels import _lax_ms_sfh_scalar_kern_sum
from ..kernels.main_sequence_kernels import _lax_ms_sfh_scalar_kern_sum as kern_sum
from ..kernels.main_sequence_kernels_tpeak import (
    _lax_ms_sfh_scalar_kern_scan as kern_scan_tpeak,
)
from ..kernels.main_sequence_kernels_tpeak import (
    _lax_ms_sfh_scalar_kern_sum as kern_sum_tpeak,
)


def test_main_sequence_kernels_lax_ms_sfh_scalar_kern_scan_vs_sum():
    lgt0 = jnp.log10(TODAY)
    t_form = 12.0
    t_table = jnp.linspace(T_TABLE_MIN, t_form, 20)

    np.testing.assert_allclose(
        _lax_ms_sfh_scalar_kern_scan(
            t_form, OLD_DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, lgt0, FB, t_table
        ),
        _lax_ms_sfh_scalar_kern_sum(
            t_form, OLD_DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, lgt0, FB, t_table
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_main_sequence_kernels_tpeak():

    t_table = jnp.logspace(0, LGT0, 100)
    t_form = jnp.logspace(0, LGT0, 30)

    sfr_sum = np.array(
        [
            kern_sum(x, OLD_DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table)
            for x in t_form
        ]
    )
    sfr_scan = np.array(
        [
            kern_scan(x, OLD_DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table)
            for x in t_form
        ]
    )
    sfr_sum_tpeak = np.array(
        [
            kern_sum_tpeak(x, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table)
            for x in t_form
        ]
    )
    sfr_scan_tpeak = np.array(
        [
            kern_scan_tpeak(x, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, LGT0, FB, t_table)
            for x in t_form
        ]
    )

    assert np.allclose(sfr_sum, sfr_sum_tpeak)
    assert np.allclose(sfr_scan, sfr_scan_tpeak)
