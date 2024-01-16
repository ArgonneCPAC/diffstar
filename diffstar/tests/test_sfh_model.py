""""""
import numpy as np
from jax import random as jran

from .. import sfh_galpop, sfh_singlegal
from ..defaults import (
    DEFAULT_MS_PARAMS,
    DEFAULT_Q_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    LGT0,
    SFR_MIN,
    DiffstarParams,
    MSParams,
    QParams,
)
from ..kernels.kernel_builders import get_sfh_from_mah_kern
from ..kernels.main_sequence_kernels import (
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params_vmap,
)
from ..kernels.quenching_kernels import (
    _get_bounded_q_params,
    _get_unbounded_q_params_vmap,
)
from ..sfh_model import calc_sfh_smh_singlegal
from .test_gas import _get_default_mah_params


def _get_all_default_params():
    ms_params, q_params = DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return lgt0, mah_params, ms_params, q_params


def _get_all_default_u_params():
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return lgt0, mah_params, u_ms_params, u_q_params


def test_sfh_singlegal_evaluates():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
    assert sfh.shape == (n_t,)


def test_calc_sfh_smh_singlegal_agrees_with_sfh_singlegal_on_defaults():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, lgt0, FB)

    sfh_params = DiffstarParams(MSParams(*ms_params), QParams(*q_params))
    sfh_new, smh_new = calc_sfh_smh_singlegal(
        sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB
    )

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
        ms_params = _get_bounded_sfr_params(*u_ms_params)
        q_params = _get_bounded_q_params(*u_q_params)
        sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, lgt0, FB)

        sfh_params = DiffstarParams(MSParams(*ms_params), QParams(*q_params))
        sfh_new, smh_new = calc_sfh_smh_singlegal(
            sfh_params, mah_params, tarr, lgt0=lgt0, fb=FB
        )

        assert np.allclose(sfh, sfh_new, rtol=1e-4)
