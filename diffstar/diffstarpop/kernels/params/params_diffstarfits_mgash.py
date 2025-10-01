""" """

# flake8: noqa

from collections import OrderedDict

# Glacticus in plus ex situ
from .params_diffstarfits_mgash_galacticus_in_plus_ex_situ import (
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash,
)
from .params_diffstarfits_mgash_galacticus_in_plus_ex_situ import (
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash,
)
from .params_diffstarfits_mgash_galacticus_in_situ import (
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash,
)
from .params_diffstarfits_mgash_galacticus_in_situ import (
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash,
)
from .params_diffstarfits_mgash_smdpl_dr1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash,
)
from .params_diffstarfits_mgash_smdpl_dr1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash,
)

# SMDPL
from .params_diffstarfits_mgash_smdpl_dr1_nomerging import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_mgash,
)
from .params_diffstarfits_mgash_smdpl_dr1_nomerging import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash,
)
from .params_diffstarfits_mgash_tng import (
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_mgash,
)
from .params_diffstarfits_mgash_tng import (
    DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_mgash,
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

DiffstarPop_Params_Diffstarfits_mgash = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_mgash,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_mgash,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_mgash,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_mgash,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_mgash,
)
DiffstarPop_UParams_Diffstarfits_mgash = OrderedDict(
    smdpl_dr1_nomerging=DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_mgash,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_mgash,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_mgash,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_mgash,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_mgash,
)
