"""
"""
from collections import OrderedDict

import numpy as np

from .kernels.main_sequence_kernels import _get_unbounded_sfr_params

DEFAULT_MS_PDICT = OrderedDict(
    lgmcrit=12.0,
    lgy_at_mcrit=-1.0,
    indx_lo=1.0,
    indx_hi=-1.0,
    tau_dep=2.0,
)
DEFAULT_MS_PARAMS = np.array(list(DEFAULT_MS_PDICT.values()))
DEFAULT_U_MS_PARAMS = _get_unbounded_sfr_params(*DEFAULT_MS_PARAMS)

DEFAULT_U_Q_PDICT = OrderedDict(
    u_lg_qt=1.0, u_lg_qs=-0.3, u_lg_drop=-1.0, u_lg_rejuv=-0.5
)
DEFAULT_U_Q_PARAMS = np.array(list(DEFAULT_U_Q_PDICT.values()))
