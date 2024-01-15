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
    _quenching_kern,
    _quenching_kern_u_params,
)


def test_default_params_imports_from_top_level():
    try:
        from .. import DEFAULT_DIFFSTAR_PARAMS  # noqa
    except ImportError:
        raise ImportError("DEFAULT_DIFFSTAR_PARAMS should import from top level")


def test_default_u_params_imports_from_top_level():
    try:
        from .. import DEFAULT_DIFFSTAR_U_PARAMS  # noqa
    except ImportError:
        raise ImportError("DEFAULT_DIFFSTAR_U_PARAMS should import from top level")


def test_get_bounded_diffstar_params_imports_from_top_level():
    try:
        from .. import get_bounded_diffstar_params  # noqa
    except ImportError:
        raise ImportError("get_bounded_diffstar_params should import from top level")


def test_get_unbounded_diffstar_params_imports_from_top_level():
    try:
        from .. import get_unbounded_diffstar_params  # noqa
    except ImportError:
        raise ImportError("get_unbounded_diffstar_params should import from top level")


def test_DiffstarUParams_imports_from_top_level():
    try:
        from .. import DiffstarUParams  # noqa
    except ImportError:
        raise ImportError("DiffstarUParams should import from top level")


def test_DiffstarParams_imports_from_top_level():
    try:
        from .. import DiffstarParams  # noqa
    except ImportError:
        raise ImportError("DiffstarParams should import from top level")


def test_default_diffstar_params():
    gen = zip(
        defaults.DEFAULT_DIFFSTAR_PARAMS._fields,
        defaults.DEFAULT_DIFFSTAR_U_PARAMS._fields,
    )
    for key, u_key in gen:
        assert "u_" + key == u_key

    assert defaults.DEFAULT_DIFFSTAR_PARAMS._fields == ("ms_params", "q_params")
    assert defaults.DEFAULT_DIFFSTAR_U_PARAMS._fields == ("u_ms_params", "u_q_params")

    assert np.allclose(
        defaults.DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params,
        np.array(list(DEFAULT_U_MS_PDICT.values())),
    )
    assert np.allclose(
        defaults.DEFAULT_DIFFSTAR_U_PARAMS.u_q_params,
        np.array(list(DEFAULT_U_Q_PDICT.values())),
    )


def test_get_bounded_diffstar_params():
    p = defaults.get_bounded_diffstar_params(defaults.DEFAULT_DIFFSTAR_U_PARAMS)
    assert np.allclose(p.ms_params, defaults.DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert np.allclose(p.q_params, defaults.DEFAULT_DIFFSTAR_PARAMS.q_params)

    u_p = defaults.get_unbounded_diffstar_params(defaults.DEFAULT_DIFFSTAR_PARAMS)
    assert np.allclose(u_p.u_ms_params, defaults.DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params)
    assert np.allclose(u_p.u_q_params, defaults.DEFAULT_DIFFSTAR_U_PARAMS.u_q_params)


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


def test_default_q_params_unquenched():
    tarr = np.linspace(0.1, 13.8, 200)
    lgtarr = np.log10(tarr)

    res = _quenching_kern(lgtarr, *defaults.DEFAULT_Q_PARAMS_UNQUENCHED)
    assert np.all(res > 0.99)

    res2 = _quenching_kern_u_params(lgtarr, *defaults.DEFAULT_Q_U_PARAMS_UNQUENCHED)
    assert np.all(res2 > 0.99)

    assert np.allclose(res, res2)
