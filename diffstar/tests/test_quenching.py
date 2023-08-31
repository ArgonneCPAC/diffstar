"""
"""
import numpy as np

from ..defaults import DEFAULT_U_Q_PARAMS
from ..quenching import quenching_function


def test_quenching_function():
    lgtarr = np.linspace(-1, 1.2, 200)

    q = quenching_function(lgtarr, *DEFAULT_U_Q_PARAMS)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0)
    assert np.all(q <= 1)
