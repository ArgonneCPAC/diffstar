"""
"""
from ...defaults import DEFAULT_MS_PDICT
from ...kernels.main_sequence_kernels import MS_PARAM_BOUNDS_PDICT


def test_sfh_parameter_bounds():
    for key, val in DEFAULT_MS_PDICT.items():
        assert MS_PARAM_BOUNDS_PDICT[key][0] < val < MS_PARAM_BOUNDS_PDICT[key][1]
