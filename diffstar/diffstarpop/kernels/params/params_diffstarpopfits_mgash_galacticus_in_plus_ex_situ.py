from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 11.990),
        ("mean_ulgm_mseq_ytp", 11.146),
        ("mean_ulgm_mseq_lo", 0.415),
        ("mean_ulgm_mseq_hi", -0.261),
        ("mean_ulgy_mseq_int", -8.857),
        ("mean_ulgy_mseq_slp", 1.175),
        ("mean_ul_mseq_int", -2.622),
        ("mean_ul_mseq_slp", 0.819),
        ("mean_uh_mseq_int", -0.577),
        ("mean_uh_mseq_slp", 0.606),
        ("mean_ulgm_qseq_xtp", 13.247),
        ("mean_ulgm_qseq_ytp", 11.923),
        ("mean_ulgm_qseq_lo", 0.290),
        ("mean_ulgm_qseq_hi", 0.500),
        ("mean_ulgy_qseq_int", -9.306),
        ("mean_ulgy_qseq_slp", 0.592),
        ("mean_ul_qseq_int", -2.997),
        ("mean_ul_qseq_slp", 3.163),
        ("mean_uh_qseq_int", -1.156),
        ("mean_uh_qseq_slp", -0.498),
        ("mean_uqt_int", 1.109),
        ("mean_uqt_slp", 0.024),
        ("mean_uqs_int", 1.581),
        ("mean_uqs_slp", 0.434),
        ("mean_udrop_int", -2.222),
        ("mean_udrop_slp", 0.983),
        ("mean_urej_int", -1.646),
        ("mean_urej_slp", 2.075),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.999),
        ("std_ulgy_mseq_int", 0.053),
        ("std_ulgy_mseq_slp", 0.999),
        ("std_ul_mseq_int", 0.054),
        ("std_ul_mseq_slp", 0.058),
        ("std_uh_mseq_int", 0.119),
        ("std_uh_mseq_slp", 0.999),
        ("std_ulgm_qseq_int", 0.011),
        ("std_ulgm_qseq_slp", -0.341),
        ("std_ulgy_qseq_int", 0.011),
        ("std_ulgy_qseq_slp", 0.198),
        ("std_ul_qseq_int", 0.825),
        ("std_ul_qseq_slp", -0.999),
        ("std_uh_qseq_int", 0.133),
        ("std_uh_qseq_slp", 0.082),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.122),
        ("std_uqt_slp", 0.048),
        ("std_uqs_int", 0.999),
        ("std_uqs_slp", 0.636),
        ("std_udrop_int", 0.435),
        ("std_udrop_slp", 0.285),
        ("std_urej_int", 0.011),
        ("std_urej_slp", 0.739),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.476),
        ("frac_quench_cen_k_tpeak", 2.282),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 12.619),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.028),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 5.944),
        ("frac_quench_sat_k_tpeak", 9.994),
        ("frac_quench_sat_x0_ylotpeak", 11.906),
        ("frac_quench_sat_x0_yhitpeak", 12.378),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.677),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.924),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 2.373),
        ("delta_uqt_k", 0.450),
        ("delta_uqt_ylo", -0.622),
        ("delta_uqt_yhi", 0.075),
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


DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key
    for key in DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS = (
    get_unbounded_diffstarpop_params(
        DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS
    )
)
