""""""

from .. import DEFAULT_DIFFSTARPOP_PARAMS
from ..kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)

DEFAULT_MODELNAME = "smdpl_dr1_nomerging"


def test_default_params_has_same_fields_as_universemachine_dr1():
    params = DiffstarPop_Params_Diffstarpopfits_mgash[DEFAULT_MODELNAME]
    gen = zip(
        DEFAULT_DIFFSTARPOP_PARAMS._fields,
        params._fields,
    )
    for default_key, um_dr1_key in gen:
        assert default_key == um_dr1_key


def test_default_params_has_same_values_as_universemachine_dr1():
    params = DiffstarPop_Params_Diffstarpopfits_mgash[DEFAULT_MODELNAME]
    gen = zip(
        DEFAULT_DIFFSTARPOP_PARAMS._fields,
        params._fields,
    )
    for default_key, um_dr1_key in gen:
        assert default_key == um_dr1_key
        default_val = getattr(DEFAULT_DIFFSTARPOP_PARAMS, default_key)
        um_dr1_val = getattr(params, um_dr1_key)
        assert default_val == um_dr1_val, default_key
