"""
"""
from ..stars import DEFAULT_SFR_PARAMS, _SFR_PARAM_BOUNDS


def test_sfh_parameter_bounds():
    for key, val in DEFAULT_SFR_PARAMS.items():
        assert _SFR_PARAM_BOUNDS[key][0] < val < _SFR_PARAM_BOUNDS[key][1]
