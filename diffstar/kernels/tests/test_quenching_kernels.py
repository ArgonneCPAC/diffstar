"""
"""
import numpy as np

from ...defaults import DEFAULT_Q_PARAMS, DEFAULT_U_Q_PARAMS, Q_PARAM_BOUNDS_PDICT
from ..quenching_kernels import (
    DEFAULT_Q_PARAMS_UNQUENCHED,
    DEFAULT_Q_U_PARAMS_UNQUENCHED,
    DEFAULT_U_Q_PDICT,
    _get_bounded_q_params,
    _get_unbounded_q_params,
    _quenching_kern,
    _quenching_kern_u_params,
)


def test_unbounding_function_returns_finite_results_on_default_q_params():
    inferred_default_u_q_params = _get_unbounded_q_params(*DEFAULT_Q_PARAMS)
    assert np.all(np.isfinite(inferred_default_u_q_params))


def test_bounding_function_returns_finite_results_on_default_u_q_params():
    inferred_default_q_params = _get_bounded_q_params(*DEFAULT_U_Q_PARAMS)
    assert np.all(np.isfinite(inferred_default_q_params))


def test_default_quenching_params_respect_bounds():
    keys = list(Q_PARAM_BOUNDS_PDICT.keys())
    correct_u_keys = ["u_" + key for key in keys]
    actual_u_keys = list(DEFAULT_U_Q_PDICT.keys())
    assert correct_u_keys == actual_u_keys

    for key, bounds in Q_PARAM_BOUNDS_PDICT.items():
        lo, hi = bounds
        default_val = DEFAULT_U_Q_PDICT["u_" + key]
        assert lo < default_val < hi


def test_unbounded_and_bounded_quenching_functions_agree():
    tarr = np.linspace(0.1, 13.8, 200)
    lgtarr = np.log10(tarr)

    q2 = _quenching_kern_u_params(lgtarr, *DEFAULT_U_Q_PARAMS)

    lg_qt, lg_lg_q_dt, lg_drop, lg_rejuv = DEFAULT_Q_PARAMS
    lg_q_dt = 10**lg_lg_q_dt
    quenching_kern_arguments = lg_qt, lg_q_dt, lg_drop, lg_rejuv
    q1 = _quenching_kern(lgtarr, *quenching_kern_arguments)
    assert np.allclose(q1, q2, rtol=1e-3)


def test_quenching_bounding_functions_correctly_invert():
    inferred_default_u_q_params = _get_unbounded_q_params(*DEFAULT_Q_PARAMS)
    inferred_default_q_params = _get_bounded_q_params(*DEFAULT_U_Q_PARAMS)
    assert np.all(np.isfinite(inferred_default_u_q_params))
    assert np.all(np.isfinite(inferred_default_q_params))
    assert np.allclose(inferred_default_u_q_params, DEFAULT_U_Q_PARAMS, rtol=1e-3)
    assert np.allclose(inferred_default_q_params, DEFAULT_Q_PARAMS, rtol=1e-3)


def test_unbounded_quenching_function_has_expected_behavior_on_default_u_params():
    tarr = np.linspace(0.1, 13.8, 200)
    lgtarr = np.log10(tarr)

    q2 = _quenching_kern_u_params(lgtarr, *DEFAULT_U_Q_PARAMS)
    assert np.all(np.isfinite(q2))
    assert np.all(q2 >= 0)
    assert np.all(q2 <= 1)
    assert np.allclose(q2[0], 1.0)
    assert np.any(q2 < 1)


def test_quenching_function_has_expected_behavior_on_default_params():
    tarr = np.linspace(0.1, 13.8, 200)
    lgtarr = np.log10(tarr)

    lg_qt, lg_lg_q_dt, lg_drop, lg_rejuv = DEFAULT_Q_PARAMS
    lg_q_dt = 10**lg_lg_q_dt
    quenching_kern_arguments = lg_qt, lg_q_dt, lg_drop, lg_rejuv

    q = _quenching_kern(lgtarr, *quenching_kern_arguments)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0)
    assert np.all(q <= 1)
    assert np.allclose(q[0], 1.0)
    assert np.any(q < 1)

    lg_qt, q_dt, lg_q_drop, lg_q_rejuv = DEFAULT_Q_PARAMS
    actual_lg_q_drop = np.log10(
        _quenching_kern(lg_qt, lg_qt, q_dt, lg_q_drop, lg_q_rejuv)
    )
    assert np.allclose(actual_lg_q_drop, lg_q_drop, rtol=1e-3)


def test_default_q_params_unquenched():
    tarr = np.linspace(0.1, 13.8, 200)
    lgtarr = np.log10(tarr)

    res = _quenching_kern(lgtarr, *DEFAULT_Q_PARAMS_UNQUENCHED)
    assert np.all(res > 0.99)

    res2 = _quenching_kern_u_params(lgtarr, *DEFAULT_Q_U_PARAMS_UNQUENCHED)
    assert np.all(res2 > 0.99)

    assert np.allclose(res, res2)
