""""""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS as OLD_DEFAULT_MAH_PARAMS
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, DiffmahParams
from jax import random as jran

from ..defaults import (
    DEFAULT_MS_PARAMS,
    DEFAULT_Q_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    LGT0,
    SFR_MIN,
    DiffstarParams,
    DiffstarUParams,
    MSParams,
    MSUParams,
    QParams,
    QUParams,
    get_bounded_diffstar_params,
)
from ..kernels.main_sequence_kernels import (
    _get_bounded_sfr_params as _old_get_bounded_sfr_params,
)
from ..kernels.quenching_kernels import (
    _get_bounded_q_params as _old_get_bounded_q_params,
)
from ..sfh import sfh_galpop, sfh_singlegal
from ..sfh_model_tpeak import calc_sfh_galpop, calc_sfh_singlegal


def _get_all_default_params():
    ms_params, q_params = DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
    return LGT0, DEFAULT_MAH_PARAMS, ms_params, q_params


def _get_all_default_u_params():
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    return LGT0, DEFAULT_MAH_PARAMS, u_ms_params, u_q_params


def test_calc_sfh_singlegal_imports_from_top_level():
    from .. import calc_sfh_singlegal as _func  # noqa


def test_calc_sfh_galpop_imports_from_top_level():
    from .. import calc_sfh_galpop as _func  # noqa


def test_sfh_singlegal_evaluates():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, OLD_DEFAULT_MAH_PARAMS, ms_params, q_params, LGT0, FB)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
    assert sfh.shape == (n_t,)


def test_calc_sfh_smh_singlegal_agrees_with_sfh_singlegal_on_defaults():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, OLD_DEFAULT_MAH_PARAMS, ms_params, q_params, lgt0, FB)

    sfh_params = DiffstarParams(MSParams(*ms_params), QParams(*q_params))
    sfh_new = calc_sfh_singlegal(sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB)

    assert np.allclose(sfh, sfh_new, rtol=1e-4)


def test_calc_sfh_smh_singlegal_agrees_with_sfh_singlegal_on_randoms():
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
        ms_params = _old_get_bounded_sfr_params(*u_ms_params)
        q_params = _old_get_bounded_q_params(*u_q_params)
        sfh = sfh_singlegal(tarr, OLD_DEFAULT_MAH_PARAMS, ms_params, q_params, lgt0, FB)

        sfh_params = DiffstarParams(MSParams(*ms_params), QParams(*q_params))
        sfh_new = calc_sfh_singlegal(sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB)

        assert np.allclose(sfh, sfh_new, rtol=1e-4)


def test_calc_sfh_smh_singlegal_agrees_with_sfh_singlegal_on_u_randoms():
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
        sfh = sfh_singlegal(
            tarr,
            OLD_DEFAULT_MAH_PARAMS,
            u_ms_params,
            u_q_params,
            lgt0,
            FB,
            ms_param_type="unbounded",
            q_param_type="unbounded",
        )
        sfh_u_params = DiffstarUParams(MSUParams(*u_ms_params), QUParams(*u_q_params))
        sfh_params = get_bounded_diffstar_params(sfh_u_params)
        sfh_new = calc_sfh_singlegal(sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB)
        assert np.allclose(sfh, sfh_new, rtol=1e-4)

        sfh_new2, smh_new2 = calc_sfh_singlegal(
            sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB, return_smh=True
        )
        assert np.allclose(sfh_new, sfh_new2)
        assert np.all(np.isfinite(smh_new2))


def test_calc_sfh_smh_galpop_agrees_with_sfh_galpop():
    n_t = 100
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    old_mah_params = np.array(OLD_DEFAULT_MAH_PARAMS).reshape((1, -1))
    ms_params = np.array(ms_params).reshape((1, -1))
    q_params = np.array(q_params).reshape((1, -1))
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_galpop(tarr, old_mah_params, ms_params, q_params)

    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    zz = np.zeros(1)
    mah_params = DiffmahParams(*[zz + p for p in mah_params])
    ms_params = MSParams(*[zz + p for p in ms_params])
    q_params = QParams(*[zz + p for p in q_params])
    sfh_params = DiffstarParams(ms_params, q_params)
    sfh_new = calc_sfh_galpop(sfh_params, mah_params, tarr)
    assert np.allclose(sfh, sfh_new)
    assert sfh_new.shape == (1, n_t)

    sfh_new2, smh_new2 = calc_sfh_galpop(sfh_params, mah_params, tarr, return_smh=True)
    assert np.allclose(sfh_new, sfh_new2)
    assert np.all(np.isfinite(smh_new2))
