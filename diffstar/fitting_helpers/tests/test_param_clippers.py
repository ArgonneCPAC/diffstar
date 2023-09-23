"""
"""
import numpy as np
from jax import random as jran

from ...kernels.main_sequence_kernels import (
    MS_PARAM_BOUNDS_PDICT,
    _get_bounded_sfr_params_vmap,
    _get_unbounded_sfr_params_vmap,
)
from ...kernels.quenching_kernels import (
    Q_PARAM_BOUNDS_PDICT,
    _get_bounded_q_params_vmap,
    _get_unbounded_q_params_vmap,
)
from ..param_clippers import _EPS, ms_param_clipper, q_param_clipper


def test_ms_param_clipper_implements_correct_bounding_behavior():
    n_gals = 1_000
    ran_key = jran.PRNGKey(0)
    u_ms_params = 10 ** jran.uniform(ran_key, minval=-5, maxval=5, shape=(n_gals, 5))

    ms_params = _get_bounded_sfr_params_vmap(u_ms_params)
    assert np.all(np.isfinite(ms_params))

    # Enforce that clipping is actually necessary for these inputs
    unclipped_u_ms_params = _get_unbounded_sfr_params_vmap(ms_params)
    assert not np.all(np.isfinite(unclipped_u_ms_params))

    clipped_ms_params = ms_param_clipper(ms_params)
    assert clipped_ms_params.shape == (n_gals, 5)
    assert np.all(np.isfinite(clipped_ms_params))
    assert np.allclose(ms_params, clipped_ms_params, atol=_EPS)

    for ip, bounds in enumerate(MS_PARAM_BOUNDS_PDICT.values()):
        lo, hi = bounds
        assert np.all(clipped_ms_params[:, ip] > lo)
        assert np.all(clipped_ms_params[:, ip] < hi)

    clipped_u_ms_params = _get_unbounded_sfr_params_vmap(clipped_ms_params)
    assert np.all(np.isfinite(clipped_u_ms_params))


def test_q_param_clipper_implements_correct_bounding_behavior():
    n_gals = 1_000
    ran_key = jran.PRNGKey(0)
    u_q_params = 10 ** jran.uniform(ran_key, minval=-5, maxval=5, shape=(n_gals, 4))

    q_params = _get_bounded_q_params_vmap(u_q_params)
    assert np.all(np.isfinite(q_params))

    # Enforce that clipping is actually necessary for these inputs
    unclipped_u_q_params = _get_unbounded_q_params_vmap(q_params)
    assert not np.all(np.isfinite(unclipped_u_q_params))

    clipped_q_params = q_param_clipper(q_params)
    assert clipped_q_params.shape == (n_gals, 4)
    assert np.all(np.isfinite(clipped_q_params))
    assert np.allclose(q_params, clipped_q_params, atol=_EPS)

    for ip, bounds in enumerate(Q_PARAM_BOUNDS_PDICT.values()):
        lo, hi = bounds
        assert np.all(clipped_q_params[:, ip] > lo)
        assert np.all(clipped_q_params[:, ip] < hi)

    assert np.all(clipped_q_params[:, 3] > clipped_q_params[:, 2])

    clipped_u_q_params = _get_unbounded_q_params_vmap(clipped_q_params)
    assert clipped_u_q_params.shape == (n_gals, 4)
    assert np.all(np.isfinite(clipped_u_q_params[:, 3]))
