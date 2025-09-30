import numpy as np
from jax import numpy as jnp, jit as jjit, grad
from ....loss_kernels.namedtuple_utils_mgash import (
    tuple_to_array,
)

from ...defaults_mgash import (
    get_unbounded_diffstarpop_params,
    get_bounded_diffstarpop_params,
)

# SMDPL
from ..params_diffstarpopfits_mgash_smdpl_dr1_nomerging import (
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS as PARAMS_SMDPL,
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS as U_PARAMS_SMDPL,
)

# SMDPL DR1
from ..params_diffstarpopfits_mgash_smdpl_dr1 import (
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as PARAMS_SMDPL_DR1,
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as U_PARAMS_SMDPL_DR1,
)

# TNG
from ..params_diffstarpopfits_mgash_tng import (
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS as PARAMS_TNG,
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS as U_PARAMS_TNG,
)

# Galacticus IN
from ..params_diffstarpopfits_mgash_galacticus_in_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as PARAMS_GALACTICUS_IN,
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as U_PARAMS_GALACTICUS_IN,
)

# Galacticus INPLUSEX
from ..params_diffstarpopfits_mgash_galacticus_in_plus_ex_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as PARAMS_GALACTICUS_INPLUSEX,
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as U_PARAMS_GALACTICUS_INPLUSEX,
)

# All simulations
from ..params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
    DiffstarPop_UParams_Diffstarpopfits_mgash,
    sim_name_list,
)


def _test_onesim(params, uparams):
    arr_params = tuple_to_array(params)
    arr_u_params = tuple_to_array(uparams)

    assert np.all(np.isfinite(arr_params)), params
    assert np.all(np.isfinite(arr_u_params)), uparams

    arr_u_params_bound = get_bounded_diffstarpop_params(uparams)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(params)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))


def test_allsims():
    for sim_name in sim_name_list:
        _params = DiffstarPop_Params_Diffstarpopfits_mgash[sim_name]
        _uparams = DiffstarPop_UParams_Diffstarpopfits_mgash[sim_name]
        _test_onesim(_params, _uparams)


def _add_params(params):
    return jnp.sum(params) ** 2


_add_params_grad = jjit(grad(_add_params, argnums=0))


def test_smdpl():
    arr_params = tuple_to_array(PARAMS_SMDPL)
    arr_u_params = tuple_to_array(U_PARAMS_SMDPL)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_SMDPL)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_SMDPL)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))


def test_smdpl_dr1():
    arr_params = tuple_to_array(PARAMS_SMDPL_DR1)
    arr_u_params = tuple_to_array(U_PARAMS_SMDPL_DR1)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_SMDPL_DR1)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_SMDPL_DR1)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))


def test_tng():
    arr_params = tuple_to_array(PARAMS_TNG)
    arr_u_params = tuple_to_array(U_PARAMS_TNG)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_TNG)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_TNG)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))


def test_galacticus_in():
    arr_params = tuple_to_array(PARAMS_GALACTICUS_IN)
    arr_u_params = tuple_to_array(U_PARAMS_GALACTICUS_IN)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_GALACTICUS_IN)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_GALACTICUS_IN)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))


def test_galacticus_inplusex():
    arr_params = tuple_to_array(PARAMS_GALACTICUS_INPLUSEX)
    arr_u_params = tuple_to_array(U_PARAMS_GALACTICUS_INPLUSEX)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_GALACTICUS_INPLUSEX)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_GALACTICUS_INPLUSEX)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)

    grad_val = _add_params_grad(arr_u_params)
    assert np.all(np.isfinite(grad_val))
