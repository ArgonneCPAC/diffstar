"""
"""
from ..stars import _SFR_PARAM_BOUNDS, DEFAULT_SFR_PARAMS


def test_sfh_parameter_bounds():
    for key, val in DEFAULT_SFR_PARAMS.items():
        assert _SFR_PARAM_BOUNDS[key][0] < val < _SFR_PARAM_BOUNDS[key][1]
