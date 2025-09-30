""" """

import typing
from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from .satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
    get_bounded_satquenchpop_params,
    get_unbounded_satquenchpop_params,
)
from .sfh_pdf_mgash import (
    SFH_PDF_QUENCH_PARAMS,
    get_bounded_sfh_pdf_params,
    get_unbounded_sfh_pdf_params,
)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array


DEFAULT_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DEFAULT_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)


@jjit
def get_bounded_diffstarpop_params(diffstarpop_u_params):
    sfh_pdf_cens_params = get_bounded_sfh_pdf_params(
        diffstarpop_u_params.u_sfh_pdf_cens_params
    )
    satquench_params = get_bounded_satquenchpop_params(
        diffstarpop_u_params.u_satquench_params
    )
    return DiffstarPopParams(sfh_pdf_cens_params, satquench_params)


@jjit
def get_unbounded_diffstarpop_params(diffstarpop_params):
    u_sfh_pdf_params = get_unbounded_sfh_pdf_params(
        diffstarpop_params.sfh_pdf_cens_params
    )
    u_satquench_params = get_unbounded_satquenchpop_params(
        diffstarpop_params.satquench_params
    )
    return DiffstarPopUParams(u_sfh_pdf_params, u_satquench_params)


DEFAULT_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DEFAULT_DIFFSTARPOP_PARAMS
)
