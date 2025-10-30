""" """

# flake8: noqa

import typing
from collections import namedtuple, OrderedDict


# SMDPL
from .params_diffstarpopfits_mgash_ecrit_qt_smdpl_dr1_nomerging import (
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

# SMDPL DR1

from .params_diffstarpopfits_mgash_ecrit_qt_smdpl_dr1 import (
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

# TNG

from .params_diffstarpopfits_mgash_ecrit_qt_tng import (
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)


# Glacticus in situ

from .params_diffstarpopfits_mgash_ecrit_qt_galacticus_in_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

# Glacticus in plus ex situ
from .params_diffstarpopfits_mgash_ecrit_qt_galacticus_in_plus_ex_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

sim_name_list = [
    "smdpl_dr1_nomerging",
    "smdpl_dr1",
    "tng",
    "galacticus_in_situ",
    "galacticus_in_plus_ex_situ",
]

DiffstarPop_Params_Diffstarpopfits_mgash_ecrit_qt = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    smdpl_dr1=DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    tng=DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    galacticus_in_situ=DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    galacticus_in_plus_ex_situ=DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
DiffstarPop_UParams_Diffstarpopfits_mgash_ecrit_qt = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    smdpl_dr1=DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    tng=DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    galacticus_in_situ=DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    galacticus_in_plus_ex_situ=DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)
