"""
"""
# flake8: noqa
import numpy as np

TODAY = 13.8
LGT0 = np.log10(TODAY)


# Constants related to SFH integrals
SFR_MIN = 1e-14
T_BIRTH_MIN = 0.001
N_T_LGSM_INTEGRATION = 100
DEFAULT_N_STEPS = 50


from .kernels.gas_consumption import FB
from .kernels.main_sequence_kernels import (
    DEFAULT_MS_PARAMS,
    DEFAULT_MS_PDICT,
    DEFAULT_U_MS_PARAMS,
    INDX_K,
)
from .kernels.quenching_kernels import (
    DEFAULT_Q_PARAMS,
    DEFAULT_U_Q_PARAMS,
    DEFAULT_U_Q_PDICT,
    Q_PARAM_BOUNDS_PDICT,
)
