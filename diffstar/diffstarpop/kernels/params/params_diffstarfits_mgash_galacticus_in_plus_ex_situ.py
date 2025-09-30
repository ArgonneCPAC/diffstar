from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.780),
        ("mean_ulgm_mseq_ytp", 11.668),
        ("mean_ulgm_mseq_lo", 0.748),
        ("mean_ulgm_mseq_hi", 0.400),
        ("mean_ulgy_mseq_int", -9.299),
        ("mean_ulgy_mseq_slp", 0.409),
        ("mean_ul_mseq_int", -0.639),
        ("mean_ul_mseq_slp", 0.988),
        ("mean_uh_mseq_int", -0.607),
        ("mean_uh_mseq_slp", -0.312),
        ("mean_ulgm_qseq_xtp", 13.252),
        ("mean_ulgm_qseq_ytp", 11.971),
        ("mean_ulgm_qseq_lo", 0.639),
        ("mean_ulgm_qseq_hi", 0.208),
        ("mean_ulgy_qseq_int", -9.243),
        ("mean_ulgy_qseq_slp", 0.660),
        ("mean_ul_qseq_int", -0.758),
        ("mean_ul_qseq_slp", 0.421),
        ("mean_uh_qseq_int", -0.661),
        ("mean_uh_qseq_slp", -0.334),
        ("mean_uqt_int", 1.065),
        ("mean_uqt_slp", -0.003),
        ("mean_uqs_int", -0.352),
        ("mean_uqs_slp", -0.273),
        ("mean_udrop_int", -1.938),
        ("mean_udrop_slp", 0.551),
        ("mean_urej_int", -0.620),
        ("mean_urej_slp", -0.206),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.355),
        ("std_ulgm_mseq_slp", -0.015),
        ("std_ulgy_mseq_int", 0.277),
        ("std_ulgy_mseq_slp", 0.103),
        ("std_ul_mseq_int", 2.221),
        ("std_ul_mseq_slp", 0.900),
        ("std_uh_mseq_int", 0.651),
        ("std_uh_mseq_slp", -0.155),
        ("std_ulgm_qseq_int", 0.313),
        ("std_ulgm_qseq_slp", -0.022),
        ("std_ulgy_qseq_int", 0.257),
        ("std_ulgy_qseq_slp", 0.075),
        ("std_ul_qseq_int", 2.241),
        ("std_ul_qseq_slp", -0.130),
        ("std_uh_qseq_int", 0.482),
        ("std_uh_qseq_slp", 0.007),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.067),
        ("std_uqt_slp", 0.013),
        ("std_uqs_int", 0.900),
        ("std_uqs_slp", 0.900),
        ("std_udrop_int", 0.589),
        ("std_udrop_slp", 0.049),
        ("std_urej_int", 0.867),
        ("std_urej_slp", 0.049),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 13.008),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.446),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.100),
        ("frac_quench_sat_x0_yhitpeak", 13.008),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.446),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.001),
        ("delta_uqt_k", 0.836),
        ("delta_uqt_ylo", -0.583),
        ("delta_uqt_yhi", -0.006),
        ("delta_uqt_slope", -0.017),
    ]
)
SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(DELTA_UQT_PDICT)

QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array


DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS = (
    get_unbounded_diffstarpop_params(
        DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS
    )
)
