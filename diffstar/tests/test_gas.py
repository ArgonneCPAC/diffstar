"""
"""
from ..kernels.gas_consumption import _get_lagged_gas
from ..kernels.main_sequence_kernels import MS_BOUNDING_SIGMOID_PDICT
from .test_diffstar_is_frozen import calc_sfh_on_default_params


def test_lagged_gas():
    args, sfh = calc_sfh_on_default_params()
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params = args

    tau_dep = 2.0
    tau_dep_max = MS_BOUNDING_SIGMOID_PDICT["tau_dep"][3]
    lagged_gas = _get_lagged_gas(lgt, dt, dmhdt, tau_dep, tau_dep_max)
    assert lagged_gas.shape == dmhdt.shape
