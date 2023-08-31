"""
"""
import numpy as np

from ..quenching import DEFAULT_Q_PARAMS, quenching_function


def test_quenching_function():
    lgtarr = np.linspace(-1, 1.2, 200)

    q = quenching_function(lgtarr, *DEFAULT_Q_PARAMS.values())
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0)
    assert np.all(q <= 1)
