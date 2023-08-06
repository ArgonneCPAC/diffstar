"""
"""
import numpy as np
from jax import random as jran

from ..utils import _get_dt_array, _jax_get_dt_array


def test_jax_get_dt_array_linspace():
    tarr = np.linspace(1, 13.8, 50)
    dtarr_np = _get_dt_array(tarr)
    dtarr_jnp = _jax_get_dt_array(tarr)
    assert np.allclose(dtarr_np, dtarr_jnp, atol=0.01)


def test_jax_get_dt_array_random():
    n_tests = 10
    ran_key = jran.PRNGKey(0)
    for __ in range(n_tests):
        ran_key, key = jran.split(ran_key, 2)
        tarr = np.sort(jran.uniform(key, minval=0, maxval=14, shape=(50,)))
        dtarr_np = _get_dt_array(tarr)
        dtarr_jnp = _jax_get_dt_array(tarr)
        assert np.allclose(dtarr_np, dtarr_jnp, atol=0.01)
