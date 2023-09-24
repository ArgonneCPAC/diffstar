"""
"""
import numpy as np

from .. import defaults
from ..kernels.main_sequence_kernels import (
    DEFAULT_U_MS_PDICT,
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
)
from ..kernels.quenching_kernels import (
    DEFAULT_U_Q_PDICT,
    _get_bounded_q_params,
    _get_unbounded_q_params,
)


def test_default_ms_params_are_frozen():
    p = defaults.DEFAULT_MS_PARAMS
    frozen_defaults = np.array((12.0, -1.0, 1.0, -1.0, 2.0))
    assert np.allclose(p, frozen_defaults)


def test_default_q_params_are_frozen():
    p = defaults.DEFAULT_Q_PARAMS
    frozen_defaults = np.array((1.0, -0.50725, -1.01773, -0.212307))
    assert np.allclose(p, frozen_defaults)


def test_default_ms_params_bounded_consistent_with_unbounded():
    p = defaults.DEFAULT_MS_PARAMS
    u_p = defaults.DEFAULT_U_MS_PARAMS
    p2 = _get_bounded_sfr_params(*u_p)
    assert np.allclose(p, p2, rtol=0.01)

    u_p2 = _get_unbounded_sfr_params(*p)
    assert np.allclose(u_p, u_p2, rtol=0.01)


def test_default_q_params_bounded_consistent_with_unbounded():
    p = defaults.DEFAULT_Q_PARAMS
    u_p = defaults.DEFAULT_U_Q_PARAMS
    p2 = _get_bounded_q_params(*u_p)
    assert np.allclose(p, p2, rtol=0.01)

    u_p2 = _get_unbounded_q_params(*p)
    assert np.allclose(u_p, u_p2, rtol=0.01)


def test_default_indx_k():
    assert defaults.INDX_K == 9.0


def test_default_q_params_consistent_with_qparam_dict():
    pars = defaults.DEFAULT_Q_PARAMS
    pdict = defaults.DEFAULT_Q_PDICT
    for p1, p2 in zip(pars, pdict.values()):
        assert np.allclose(p1, p2)


def test_default_ms_params_consistent_with_ms_pdict():
    pars = defaults.DEFAULT_MS_PARAMS
    pdict = defaults.DEFAULT_MS_PDICT
    for p1, p2 in zip(pars, pdict.values()):
        assert np.allclose(p1, p2)


def test_default_u_q_params_consistent_with_u_q_pict():
    pars = defaults.DEFAULT_U_Q_PARAMS
    pdict = DEFAULT_U_Q_PDICT
    for p1, p2 in zip(pars, pdict.values()):
        assert np.allclose(p1, p2)


def test_default_u_ms_params_consistent_with_u_ms_pdict():
    pars = defaults.DEFAULT_U_MS_PARAMS
    pdict = DEFAULT_U_MS_PDICT
    for p1, p2 in zip(pars, pdict.values()):
        assert np.allclose(p1, p2)


def test_default_ms_params_respects_pbounds():
    pars = defaults.DEFAULT_MS_PARAMS
    bdict = defaults.MS_PARAM_BOUNDS_PDICT
    for p, bounds in zip(pars, bdict.values()):
        lo, hi = bounds
        assert lo < p < hi


def test_default_q_params_respects_pbounds():
    pars = defaults.DEFAULT_Q_PARAMS
    bdict = defaults.Q_PARAM_BOUNDS_PDICT
    for p, bounds in zip(pars, bdict.values()):
        lo, hi = bounds
        assert lo < p < hi
