"""
"""
# flake8: noqa
from collections import namedtuple

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS, DEFAULT_MAH_PDICT
from jax import jit as jjit

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
    MS_PARAM_BOUNDS_PDICT,
    MSParams,
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
)
from .kernels.quenching_kernels import (
    DEFAULT_Q_PARAMS,
    DEFAULT_Q_PARAMS_UNQUENCHED,
    DEFAULT_Q_PDICT,
    DEFAULT_Q_U_PARAMS_UNQUENCHED,
    DEFAULT_U_Q_PARAMS,
    Q_PARAM_BOUNDS_PDICT,
    QParams,
    QUParams,
    _get_bounded_q_params,
    _get_unbounded_q_params,
)

pnames = ["ms_params", "q_params"]
DiffstarParams = namedtuple("DiffstarParams", pnames)
DEFAULT_DIFFSTAR_PARAMS = DiffstarParams(DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS)

DiffstarUParams = namedtuple("DiffstarUParams", ["u_" + key for key in pnames])
DEFAULT_DIFFSTAR_U_PARAMS = DiffstarUParams(DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS)


@jjit
def get_bounded_diffstar_params(diffstar_u_params):
    ms_params = MSParams(*_get_bounded_sfr_params(*diffstar_u_params.u_ms_params))
    q_params = QParams(*_get_bounded_q_params(*diffstar_u_params.u_q_params))
    return DiffstarParams(ms_params, q_params)


@jjit
def get_unbounded_diffstar_params(diffstar_params):
    ms_params = MSParams(*_get_unbounded_sfr_params(*diffstar_params.ms_params))
    q_params = QUParams(*_get_unbounded_q_params(*diffstar_params.q_params))
    return DiffstarUParams(ms_params, q_params)
