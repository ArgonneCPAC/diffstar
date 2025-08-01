""""""

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import random as jran

from ..defaults import (
    DEFAULT_MS_PARAMS,
    DEFAULT_Q_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    LGT0,
    DiffstarUParams,
    MSUParams,
    QUParams,
    get_bounded_diffstar_params,
)
from ..mgas_model import calc_mgas_singlegal


def _get_all_default_params():
    ms_params, q_params = DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
    return LGT0, DEFAULT_MAH_PARAMS, ms_params, q_params


def _get_all_default_u_params():
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    return LGT0, DEFAULT_MAH_PARAMS, u_ms_params, u_q_params


def test_calc_mgas_singlegal_imports_from_top_level():
    from .. import calc_mgas_singlegal as _func  # noqa


def test_calc_mgas_galpop_imports_from_top_level():
    from .. import calc_mgas_galpop as _func  # noqa


def test_sfh_singlegal_evaluates_on_wide_param_range():
    lgt0, mah_params, u_ms_params_init, u_q_params_init = _get_all_default_u_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)

    ran_key = jran.PRNGKey(0)
    ntests = 20
    ran_keys = jran.split(ran_key, ntests)
    for test_key in ran_keys:
        ms_key, q_key = jran.split(test_key, 2)
        u_ms_params = jran.normal(ms_key, shape=(5,)) + np.array(u_ms_params_init)
        u_q_params = jran.normal(q_key, shape=(4,)) + np.array(u_q_params_init)
        sfh_u_params = DiffstarUParams(MSUParams(*u_ms_params), QUParams(*u_q_params))
        sfh_params = get_bounded_diffstar_params(sfh_u_params)
        res = calc_mgas_singlegal(sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB)
        assert np.all(np.isfinite(res.sfh))
        assert np.all(np.isfinite(res.smh))
        assert np.all(np.isfinite(res.dmgash))
        assert np.all(np.isfinite(res.mgash))

        res2 = calc_mgas_singlegal(sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB)

        assert np.all(np.isfinite(res2.sfh))
        assert np.all(np.isfinite(res2.smh))
        assert np.all(np.isfinite(res2.dmgash))
        assert np.all(np.isfinite(res2.mgash))
        assert np.allclose(res.sfh, res2.sfh)
        assert np.allclose(res.mgash, res2.mgash)
