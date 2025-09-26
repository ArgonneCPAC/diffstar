""" """

import numpy as np
import pytest

from .. import defaults_flat as defaults
from ..kernels.main_sequence_kernels_flat import (
    DEFAULT_MS_PDICT,
    DEFAULT_U_MS_PDICT,
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
)
from ..kernels.quenching_kernels_flat import (
    DEFAULT_Q_PDICT,
    DEFAULT_U_Q_PDICT,
    _get_bounded_q_params,
    _get_unbounded_q_params,
    _quenching_kern,
    _quenching_kern_u_params,
)

try:
    import dsps

    HAS_DSPS = True
except ImportError:
    HAS_DSPS = False

MSG_HAS_DSPS = "Must have dsps installed to run this test"


def test_get_bounded_diffstar_params_return_unbounded_namedtuple():
    diffstar_params = defaults.get_bounded_diffstar_params(
        defaults.DEFAULT_DIFFSTAR_U_PARAMS
    )
    assert diffstar_params._fields == (
        *DEFAULT_MS_PDICT.keys(),
        *DEFAULT_Q_PDICT.keys(),
    )


def test_get_unbounded_diffstar_params_return_unbounded_namedtuple():
    u_diffstar_params = defaults.get_unbounded_diffstar_params(
        defaults.DEFAULT_DIFFSTAR_PARAMS
    )
    assert u_diffstar_params._fields == (
        *DEFAULT_U_MS_PDICT.keys(),
        *DEFAULT_U_Q_PDICT.keys(),
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


def test_get_bounded_diffstar_params():
    diffstar_params = defaults.get_bounded_diffstar_params(
        defaults.DEFAULT_DIFFSTAR_U_PARAMS
    )
    assert np.allclose(diffstar_params, defaults.DEFAULT_DIFFSTAR_PARAMS)

    diffstar_u_params = defaults.get_unbounded_diffstar_params(
        defaults.DEFAULT_DIFFSTAR_PARAMS
    )
    assert np.allclose(diffstar_u_params, defaults.DEFAULT_DIFFSTAR_U_PARAMS)


def test_default_ms_params_are_frozen():
    p = defaults.DEFAULT_MS_PARAMS
    frozen_defaults = np.array((12.0, -10.0, 1.0, -1.0))
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


@pytest.mark.skipif(not HAS_DSPS, reason=MSG_HAS_DSPS)
def test_consistency_with_dsps_defaults():
    assert np.allclose(defaults.SFR_MIN, dsps.constants.SFR_MIN)
    assert np.allclose(defaults.T_TABLE_MIN, dsps.constants.T_TABLE_MIN)


def test_consistency_with_diffmah_defaults():
    from diffmah import defaults as diffmah_defaults

    assert np.allclose(defaults.LGT0, diffmah_defaults.LGT0)
