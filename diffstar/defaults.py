"""
"""
# flake8: noqa

from collections import OrderedDict

import numpy as np

from .kernels.main_sequence_kernels import (
    DEFAULT_MS_PARAMS,
    DEFAULT_MS_PDICT,
    DEFAULT_U_MS_PARAMS,
    INDX_K,
)

DEFAULT_U_Q_PDICT = OrderedDict(
    u_lg_qt=1.0, u_lg_qs=-0.3, u_lg_drop=-1.0, u_lg_rejuv=-0.5
)
DEFAULT_U_Q_PARAMS = np.array(list(DEFAULT_U_Q_PDICT.values()))
