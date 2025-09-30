from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.228),
        ("mean_ulgm_mseq_ytp", 11.925),
        ("mean_ulgm_mseq_lo", 0.733),
        ("mean_ulgm_mseq_hi", -0.212),
        ("mean_ulgy_mseq_int", -9.361),
        ("mean_ulgy_mseq_slp", 1.696),
        ("mean_ul_mseq_int", -2.997),
        ("mean_ul_mseq_slp", 0.021),
        ("mean_uh_mseq_int", -3.647),
        ("mean_uh_mseq_slp", 1.908),
        ("mean_ulgm_qseq_xtp", 11.405),
        ("mean_ulgm_qseq_ytp", 11.834),
        ("mean_ulgm_qseq_lo", 2.487),
        ("mean_ulgm_qseq_hi", 0.225),
        ("mean_ulgy_qseq_int", -9.925),
        ("mean_ulgy_qseq_slp", 0.626),
        ("mean_ul_qseq_int", -2.999),
        ("mean_ul_qseq_slp", 0.002),
        ("mean_uh_qseq_int", -1.603),
        ("mean_uh_qseq_slp", 0.194),
        ("mean_uqt_int", 1.245),
        ("mean_uqt_slp", -0.030),
        ("mean_uqs_int", 0.688),
        ("mean_uqs_slp", 1.099),
        ("mean_udrop_int", -2.064),
        ("mean_udrop_slp", -0.114),
        ("mean_urej_int", -9.313),
        ("mean_urej_slp", -11.160),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.047),
        ("std_ulgm_mseq_slp", -0.062),
        ("std_ulgy_mseq_int", 0.213),
        ("std_ulgy_mseq_slp", 0.154),
        ("std_ul_mseq_int", 0.025),
        ("std_ul_mseq_slp", 0.023),
        ("std_uh_mseq_int", 0.498),
        ("std_uh_mseq_slp", -0.443),
        ("std_ulgm_qseq_int", 0.300),
        ("std_ulgm_qseq_slp", -0.252),
        ("std_ulgy_qseq_int", 0.016),
        ("std_ulgy_qseq_slp", -0.186),
        ("std_ul_qseq_int", 0.018),
        ("std_ul_qseq_slp", -0.003),
        ("std_uh_qseq_int", 0.744),
        ("std_uh_qseq_slp", -0.314),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.033),
        ("std_uqt_slp", 0.063),
        ("std_uqs_int", 0.013),
        ("std_uqs_slp", -0.154),
        ("std_udrop_int", 0.652),
        ("std_udrop_slp", -0.998),
        ("std_urej_int", 0.203),
        ("std_urej_slp", -0.633),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 10.626),
        ("frac_quench_cen_k_tpeak", 9.976),
        ("frac_quench_cen_x0_ylotpeak", 11.006),
        ("frac_quench_cen_x0_yhitpeak", 12.101),
        ("frac_quench_cen_ylo_ylotpeak", 0.999),
        ("frac_quench_cen_ylo_yhitpeak", 0.001),
        ("frac_quench_cen_k", 1.896),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 5.829),
        ("frac_quench_sat_k_tpeak", 9.814),
        ("frac_quench_sat_x0_ylotpeak", 11.418),
        ("frac_quench_sat_x0_yhitpeak", 11.258),
        ("frac_quench_sat_ylo_ylotpeak", 0.998),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.997),
        ("frac_quench_sat_yhi", 0.922),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 10.958),
        ("delta_uqt_k", 0.092),
        ("delta_uqt_ylo", -0.594),
        ("delta_uqt_yhi", 0.285),
        ("delta_uqt_slope", -0.072),
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


DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
