"""
"""
# flake8: noqa
from collections import OrderedDict, namedtuple

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS

TODAY = 13.8
LGT0 = np.log10(TODAY)


def _get_pdict_from_namedtuple(params):
    return OrderedDict([(key, val) for key, val in zip(params._fields, params)])


DEFAULT_MAH_PDICT = _get_pdict_from_namedtuple(DEFAULT_MAH_PARAMS)


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
)
from .kernels.quenching_kernels import (
    DEFAULT_Q_PARAMS,
    DEFAULT_Q_PDICT,
    DEFAULT_U_Q_PARAMS,
    Q_PARAM_BOUNDS_PDICT,
)

pnames = [*DEFAULT_MS_PARAMS._fields, *DEFAULT_Q_PARAMS._fields]
DiffstarParams = namedtuple("DiffstarParams", pnames)
DEFAULT_DIFFSTAR_PARAMS = DiffstarParams(*DEFAULT_MS_PARAMS, *DEFAULT_Q_PARAMS)

DiffstarUParams = namedtuple("DiffstarUParams", ["u_" + key for key in pnames])
DEFAULT_DIFFSTAR_U_PARAMS = DiffstarUParams(*DEFAULT_U_MS_PARAMS, *DEFAULT_U_Q_PARAMS)
