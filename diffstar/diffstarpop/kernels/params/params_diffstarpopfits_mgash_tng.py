from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.142),
        ("mean_ulgm_mseq_ytp", 11.376),
        ("mean_ulgm_mseq_lo", 0.489),
        ("mean_ulgm_mseq_hi", 0.279),
        ("mean_ulgy_mseq_int", -9.181),
        ("mean_ulgy_mseq_slp", 1.319),
        ("mean_ul_mseq_int", 0.442),
        ("mean_ul_mseq_slp", 1.391),
        ("mean_uh_mseq_int", -2.218),
        ("mean_uh_mseq_slp", -1.589),
        ("mean_ulgm_qseq_xtp", 13.299),
        ("mean_ulgm_qseq_ytp", 11.894),
        ("mean_ulgm_qseq_lo", 0.036),
        ("mean_ulgm_qseq_hi", 0.361),
        ("mean_ulgy_qseq_int", -9.558),
        ("mean_ulgy_qseq_slp", 0.568),
        ("mean_ul_qseq_int", -2.997),
        ("mean_ul_qseq_slp", -1.791),
        ("mean_uh_qseq_int", -1.426),
        ("mean_uh_qseq_slp", -0.129),
        ("mean_uqt_int", 0.983),
        ("mean_uqt_slp", -0.358),
        ("mean_uqs_int", 1.406),
        ("mean_uqs_slp", -0.414),
        ("mean_udrop_int", -2.997),
        ("mean_udrop_slp", 1.363),
        ("mean_urej_int", -9.691),
        ("mean_urej_slp", -0.406),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.037),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", 0.001),
        ("std_ul_mseq_int", 0.123),
        ("std_ul_mseq_slp", 0.346),
        ("std_uh_mseq_int", 0.054),
        ("std_uh_mseq_slp", -0.999),
        ("std_ulgm_qseq_int", 0.015),
        ("std_ulgm_qseq_slp", -0.435),
        ("std_ulgy_qseq_int", 0.036),
        ("std_ulgy_qseq_slp", -0.103),
        ("std_ul_qseq_int", 0.161),
        ("std_ul_qseq_slp", -0.466),
        ("std_uh_qseq_int", 0.577),
        ("std_uh_qseq_slp", -0.268),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.066),
        ("std_uqt_slp", -0.002),
        ("std_uqs_int", 0.353),
        ("std_uqs_slp", -0.253),
        ("std_udrop_int", 0.011),
        ("std_udrop_slp", 0.662),
        ("std_urej_int", 0.100),
        ("std_urej_slp", -0.074),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 10.224),
        ("frac_quench_cen_k_tpeak", 2.036),
        ("frac_quench_cen_x0_ylotpeak", 13.985),
        ("frac_quench_cen_x0_yhitpeak", 12.398),
        ("frac_quench_cen_ylo_ylotpeak", 0.999),
        ("frac_quench_cen_ylo_yhitpeak", 0.234),
        ("frac_quench_cen_k", 4.291),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 10.161),
        ("frac_quench_sat_k_tpeak", 9.993),
        ("frac_quench_sat_x0_ylotpeak", 13.082),
        ("frac_quench_sat_x0_yhitpeak", 12.443),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.002),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.843),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 3.678),
        ("delta_uqt_k", 0.554),
        ("delta_uqt_ylo", -0.605),
        ("delta_uqt_yhi", 0.050),
        ("delta_uqt_slope", 0.020),
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


DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS
)
