"""
"""
import numpy as np
from diffmah.individual_halo_assembly import (
    DEFAULT_MAH_PARAMS,
    _calc_halo_history,
    _get_early_late,
)

from ..defaults import DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, LGT0
from ..kernels.gas_consumption import _get_lagged_gas
from ..kernels.main_sequence_kernels import MS_BOUNDING_SIGMOID_PDICT
from ..sfh import sfh_singlegal
from ..utils import _get_dt_array

DEFAULT_LOGM0 = 12.0


def _get_default_mah_params():
    """Return (logt0, logmp, logtc, k, early, late)"""
    mah_logtc, __, mah_ue, mah_ul = list(DEFAULT_MAH_PARAMS.values())
    early_index, late_index = _get_early_late(mah_ue, mah_ul)
    logmp = DEFAULT_LOGM0
    k = DEFAULT_MAH_PARAMS["mah_k"]
    return LGT0, logmp, mah_logtc, k, early_index, late_index


def test_lagged_gas():
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    tarr = np.linspace(0.1, 10**LGT0, 100)
    dtarr = _get_dt_array(tarr)
    lgtarr = np.log10(tarr)
    sfh_singlegal(tarr, mah_params, DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS)
    tau_dep = 2.0
    tau_dep_max = MS_BOUNDING_SIGMOID_PDICT["tau_dep"][3]
    dmhdt, log_mah = _calc_halo_history(lgtarr, *all_mah_params)
    lagged_gas = _get_lagged_gas(lgtarr, dtarr, dmhdt, tau_dep, tau_dep_max)
    assert lagged_gas.shape == dmhdt.shape
