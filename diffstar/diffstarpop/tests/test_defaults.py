""""""

from .. import DEFAULT_DIFFSTARPOP_PARAMS
from ..kernels.params.params_diffstarpopfits_mgash_smdpl_dr1 import (
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS,
)


def test_default_params_has_same_fields_as_universemachine_dr1():
    gen = zip(
        DEFAULT_DIFFSTARPOP_PARAMS._fields,
        DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields,
    )
    for default_key, um_dr1_key in gen:
        assert default_key == um_dr1_key


def test_default_params_has_same_values_as_universemachine_dr1():
    gen = zip(
        DEFAULT_DIFFSTARPOP_PARAMS._fields,
        DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields,
    )
    for default_key, um_dr1_key in gen:
        assert default_key == um_dr1_key
        default_val = getattr(DEFAULT_DIFFSTARPOP_PARAMS, default_key)
        um_dr1_val = getattr(DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS, um_dr1_key)
        assert default_val == um_dr1_val, default_key
