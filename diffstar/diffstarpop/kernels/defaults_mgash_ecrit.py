""" """

from collections import namedtuple

from jax import jit as jjit

from .satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
    DEFAULT_SATQUENCHPOP_U_PARAMS,
    get_bounded_satquenchpop_params,
    get_unbounded_satquenchpop_params,
)
from .sfh_pdf_mgash_ecrit import (
    SFH_PDF_QUENCH_PARAMS,
    SFH_PDF_QUENCH_U_PARAMS,
    get_bounded_sfh_pdf_params,
    get_unbounded_sfh_pdf_params,
)

_PDICT = SFH_PDF_QUENCH_PARAMS._asdict()
_PDICT.update(DEFAULT_SATQUENCHPOP_PARAMS._asdict())
DiffstarPopParams = namedtuple("DiffstarPopParams", list(_PDICT.keys()))

DEFAULT_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DEFAULT_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)


@jjit
def get_bounded_diffstarpop_params(diffstarpop_u_params):
    u_sfh_pdf_cens_params = [
        getattr(diffstarpop_u_params, x) for x in SFH_PDF_QUENCH_U_PARAMS._fields
    ]
    u_sfh_pdf_cens_params = SFH_PDF_QUENCH_U_PARAMS._make(u_sfh_pdf_cens_params)
    sfh_pdf_cens_params = get_bounded_sfh_pdf_params(u_sfh_pdf_cens_params)

    u_satquench_params = [
        getattr(diffstarpop_u_params, x) for x in DEFAULT_SATQUENCHPOP_U_PARAMS._fields
    ]
    u_satquench_params = DEFAULT_SATQUENCHPOP_U_PARAMS._make(u_satquench_params)

    satquench_params = get_bounded_satquenchpop_params(u_satquench_params)
    return DiffstarPopParams(*sfh_pdf_cens_params, *satquench_params)


@jjit
def get_unbounded_diffstarpop_params(diffstarpop_params):
    sfh_pdf_cens_params = [
        getattr(diffstarpop_params, x) for x in SFH_PDF_QUENCH_PARAMS._fields
    ]
    sfh_pdf_cens_params = SFH_PDF_QUENCH_PARAMS._make(sfh_pdf_cens_params)

    u_sfh_pdf_params = get_unbounded_sfh_pdf_params(sfh_pdf_cens_params)

    satquench_params = [
        getattr(diffstarpop_params, x) for x in DEFAULT_SATQUENCHPOP_PARAMS._fields
    ]
    satquench_params = DEFAULT_SATQUENCHPOP_PARAMS._make(satquench_params)
    u_satquench_params = get_unbounded_satquenchpop_params(satquench_params)

    return DiffstarPopUParams(*u_sfh_pdf_params, *u_satquench_params)


DEFAULT_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DEFAULT_DIFFSTARPOP_PARAMS
)
