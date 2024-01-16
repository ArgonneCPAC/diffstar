"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...defaults import (
    DEFAULT_DIFFSTAR_PARAMS,
    DEFAULT_DIFFSTAR_U_PARAMS,
    FB,
    LGT0,
    DiffstarUParams,
    MSUParams,
    QUParams,
    get_bounded_diffstar_params,
    get_unbounded_diffstar_params,
)
from ..history_kernel_builders import build_sfh_from_mah_kernel
from ..kernel_builders import get_sfh_from_mah_kern


def test_get_sfh_from_mah_kern_imports_from_top_level_kernels():
    from ...kernels import get_sfh_from_mah_kern

    get_sfh_from_mah_kern()


def test_new_old_kernel_builders_agree_on_defaults():
    old_sfh_kern = get_sfh_from_mah_kern()
    new_sfh_kern = build_sfh_from_mah_kernel()

    t = 10.0
    old_args = t, DEFAULT_MAH_PARAMS, *DEFAULT_DIFFSTAR_U_PARAMS, LGT0, FB
    old_sfh = old_sfh_kern(*old_args)

    new_args = t, DEFAULT_MAH_PARAMS, *DEFAULT_DIFFSTAR_PARAMS, LGT0, FB
    new_sfh = new_sfh_kern(*new_args)

    assert np.allclose(new_sfh, old_sfh, rtol=1e-3)


def test_new_old_kernel_builders_agree_on_random_u_params():
    ran_key = jran.PRNGKey(0)
    old_sfh_kern = get_sfh_from_mah_kern()
    new_sfh_kern = build_sfh_from_mah_kernel()

    ntests = 100
    ran_keys = jran.split(ran_key, ntests)
    for test_key in ran_keys:
        time_key, params_key = jran.split(test_key, 2)
        t = jran.uniform(time_key, minval=1, maxval=13.5, shape=())

        uran_params = jran.normal(params_key, shape=(9,)) * 0.1
        u_ms_params = np.array(DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params) + uran_params[:5]
        u_q_params = np.array(DEFAULT_DIFFSTAR_U_PARAMS.u_q_params) + uran_params[5:]
        old_args = t, DEFAULT_MAH_PARAMS, u_ms_params, u_q_params, LGT0, FB
        old_sfh = old_sfh_kern(*old_args)

        sfh_u_params = DiffstarUParams(MSUParams(*u_ms_params), QUParams(*u_q_params))
        sfh_params = get_bounded_diffstar_params(sfh_u_params)

        # Test u_params correctly inverts
        sfh_u_params2 = get_unbounded_diffstar_params(sfh_params)
        assert np.allclose(u_ms_params, sfh_u_params2.u_ms_params)
        assert np.allclose(u_q_params, sfh_u_params2.u_q_params)

        new_args = (
            t,
            DEFAULT_MAH_PARAMS,
            sfh_params.ms_params,
            sfh_params.q_params,
            LGT0,
            FB,
        )
        new_sfh = new_sfh_kern(*new_args)
        assert np.allclose(new_sfh, old_sfh, atol=0.01)
