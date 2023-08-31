"""
"""
import numpy as np

from .. import defaults
from ..kernels.main_sequence_kernels import (
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
)


def test_default_ms_params_are_frozen():
    p = defaults.DEFAULT_MS_PARAMS
    frozen_defaults = np.array((12.0, -1.0, 1.0, -1.0, 2.0))
    assert np.allclose(p, frozen_defaults)


def test_default_ms_params_bounded_consistent_with_unbounded():
    p = defaults.DEFAULT_MS_PARAMS
    u_p = defaults.DEFAULT_U_MS_PARAMS
    p2 = _get_bounded_sfr_params(*u_p)
    assert np.allclose(p, p2, rtol=0.01)

    u_p2 = _get_unbounded_sfr_params(*p)
    assert np.allclose(u_p, u_p2, rtol=0.01)


def test_default_indx_k():
    assert defaults.INDX_K == 9.0
