""" """

import numpy as np

from ..defaults_mgash import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DIFFSTARPOP_PBOUNDS,
    get_bounded_diffstarpop_params,
    get_unbounded_diffstarpop_params,
)


def test_get_bounded_diffstarpop_params():
    params = get_bounded_diffstarpop_params(DEFAULT_DIFFSTARPOP_U_PARAMS)
    u_params = get_unbounded_diffstarpop_params(DEFAULT_DIFFSTARPOP_PARAMS)

    for p, p2 in zip(DEFAULT_DIFFSTARPOP_PARAMS, params):
        assert np.allclose(p, p2, rtol=5e-4)

    for u_p, u_p2 in zip(DEFAULT_DIFFSTARPOP_U_PARAMS, u_params):
        assert np.allclose(u_p, u_p2, rtol=5e-4)


def test_default_values_within_bounds():
    values = DEFAULT_DIFFSTARPOP_PARAMS._asdict()
    bounds = DIFFSTARPOP_PBOUNDS._asdict()

    assert len(values) == len(bounds)

    for param_name, val in values.items():
        low, high = bounds[param_name]
        assert low <= val <= high
