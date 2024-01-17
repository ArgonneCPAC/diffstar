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
T_TABLE_MIN = 0.01
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
    MSUParams,
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
    """Calculate diffstar parameters from unbounded counterparts.

    The returned diffstar_params is the input expected by diffstar.calc_sfh_singlegal
    and diffstar.calc_sfh_galpop.

    Parameters
    ----------
    diffstar_u_params : namedtuple, length 2
        DiffstarUParams = u_ms_params, u_q_params
            u_ms_params and u_q_params are tuples of floats or ndarrays
            u_ms_params = u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep
            u_q_params = u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv

    Returns
    -------
    diffstar_params : namedtuple, length 2
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats or ndarrays
            ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    """
    ms_params = MSParams(*_get_bounded_sfr_params(*diffstar_u_params.u_ms_params))
    q_params = QParams(*_get_bounded_q_params(*diffstar_u_params.u_q_params))
    return DiffstarParams(ms_params, q_params)


@jjit
def get_unbounded_diffstar_params(diffstar_params):
    """Calculate unbounded diffstar parameters from standard params.

    This is the inverse function to get_bounded_diffstar_params

    Parameters
    ----------
    diffstar_params : namedtuple, length 2
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats or ndarrays
            ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    Returns
    -------
    diffstar_u_params : namedtuple, length 2
        DiffstarUParams = u_ms_params, u_q_params
            u_ms_params and u_q_params are tuples of floats or ndarrays
            u_ms_params = u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep
            u_q_params = u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv

    """
    u_ms_params = MSUParams(*_get_unbounded_sfr_params(*diffstar_params.ms_params))
    u_q_params = QUParams(*_get_unbounded_q_params(*diffstar_params.q_params))
    return DiffstarUParams(u_ms_params, u_q_params)
