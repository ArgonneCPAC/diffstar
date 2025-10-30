""" """

# flake8: noqa

from collections import OrderedDict

# Glacticus in plus ex situ
from .params_diffstarfits_mgash_ecrit_qt_galacticus_in_plus_ex_situ import (
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_galacticus_in_plus_ex_situ import (
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_galacticus_in_situ import (
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_galacticus_in_situ import (
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_smdpl_dr1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_smdpl_dr1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

# SMDPL
from .params_diffstarfits_mgash_ecrit_qt_smdpl_dr1_nomerging import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_smdpl_dr1_nomerging import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_tng import (
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
from .params_diffstarfits_mgash_ecrit_qt_tng import (
    DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)

# SMDPL DR1


# TNG


# Glacticus in situ


sim_name_list = [
    "smdpl_dr1_nomerging",
    "smdpl_dr1",
    "tng",
    "galacticus_in_situ",
    "galacticus_in_plus_ex_situ",
]

DiffstarPop_Params_Diffstarfits_mgash_ecrit_qt = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash_ecrit_qt,
)
DiffstarPop_UParams_Diffstarfits_mgash_ecrit_qt = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash_ecrit_qt,
)
