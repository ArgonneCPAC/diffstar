"""
"""

import numpy as np
import pytest
from jax import random as jran

from .. import utils
from ..defaults import T_TABLE_MIN

try:
    import dsps

    HAS_DSPS = True
except ImportError:
    HAS_DSPS = False

try:
    from scipy.integrate import trapezoid

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MSG_HAS_DSPS = "Must have dsps installed to run this test"
MSG_HAS_SCIPY = "Must have scipy installed to run this test"


def test_jax_get_dt_array_linspace():
    tarr = np.linspace(1, 13.8, 50)
    dtarr_np = utils._get_dt_array(tarr)
    dtarr_jnp = utils._jax_get_dt_array(tarr)
    assert np.allclose(dtarr_np, dtarr_jnp, atol=0.01)


def test_jax_get_dt_array_random():
    n_tests = 10
    ran_key = jran.PRNGKey(0)
    for __ in range(n_tests):
        ran_key, key = jran.split(ran_key, 2)
        tarr = np.sort(jran.uniform(key, minval=0, maxval=14, shape=(50,)))
        dtarr_np = utils._get_dt_array(tarr)
        dtarr_jnp = utils._jax_get_dt_array(tarr)
        assert np.allclose(dtarr_np, dtarr_jnp, atol=0.01)


@pytest.mark.skipif(not HAS_SCIPY, reason=MSG_HAS_SCIPY)
def test_cumtrapz():
    ran_key = jran.PRNGKey(0)
    n_x = 100
    n_tests = 10
    for __ in range(n_tests):
        x_key, y_key, ran_key = jran.split(ran_key, 3)
        xarr = np.sort(jran.uniform(x_key, minval=0, maxval=1, shape=(n_x,)))
        yarr = jran.uniform(y_key, minval=0, maxval=1, shape=(n_x,))
        jax_result = utils.cumtrapz(xarr, yarr)
        np_result = [trapezoid(yarr[:-i], x=xarr[:-i]) for i in range(1, n_x)][::-1]
        assert np.allclose(jax_result[:-1], np_result, rtol=1e-4)
        assert np.allclose(jax_result[-1], trapezoid(yarr, x=xarr), rtol=1e-4)


def test_cumulative_mstar_formed_returns_reasonable_arrays():
    t_table = np.linspace(T_TABLE_MIN, 13.8, 200)
    sfh_table = np.random.uniform(0, 1, t_table.size)
    smh_table = utils.cumulative_mstar_formed(t_table, sfh_table)
    assert smh_table.shape == t_table.shape
    assert np.all(smh_table > 0)
    assert np.all(np.diff(smh_table) > 0)


@pytest.mark.skipif(not HAS_DSPS, reason=MSG_HAS_DSPS)
def test_cumulative_mstar_formed_agrees_with_dsps():
    nt = 200
    t_table = np.linspace(T_TABLE_MIN, 13.8, nt)
    ran_key = jran.PRNGKey(0)
    n_tests = 10
    ran_keys = jran.split(ran_key, n_tests)
    for key in ran_keys:
        sfh_table = jran.uniform(key, minval=0, maxval=1, shape=(nt,))
        smh_table_diffstar = utils.cumulative_mstar_formed(t_table, sfh_table)
        smh_table_dsps = dsps.utils.cumulative_mstar_formed(t_table, sfh_table)
        assert np.allclose(smh_table_diffstar, smh_table_dsps, rtol=1e-4)


def test_cumulative_mstar_formed_vmap():

    n_t = 200
    t_table = np.linspace(T_TABLE_MIN, 13.8, n_t)
    ran_key = jran.PRNGKey(0)

    n_gals = 25
    sfh_table_galpop = jran.uniform(ran_key, minval=0, maxval=1, shape=(n_gals, n_t))
    smh_table_galpop = utils.cumulative_mstar_formed_galpop(t_table, sfh_table_galpop)

    for ig in range(n_gals):
        smh_table_ig = utils.cumulative_mstar_formed(t_table, sfh_table_galpop[ig, :])
        assert np.allclose(smh_table_ig, smh_table_galpop[ig, :], rtol=1e-5)
