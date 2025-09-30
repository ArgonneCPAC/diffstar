from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.126),
        ("mean_ulgm_mseq_ytp", 11.925),
        ("mean_ulgm_mseq_lo", 0.809),
        ("mean_ulgm_mseq_hi", -0.045),
        ("mean_ulgy_mseq_int", -9.342),
        ("mean_ulgy_mseq_slp", 1.631),
        ("mean_ul_mseq_int", 3.450),
        ("mean_ul_mseq_slp", 12.619),
        ("mean_uh_mseq_int", -4.995),
        ("mean_uh_mseq_slp", 0.424),
        ("mean_ulgm_qseq_xtp", 12.547),
        ("mean_ulgm_qseq_ytp", 12.283),
        ("mean_ulgm_qseq_lo", 0.612),
        ("mean_ulgm_qseq_hi", -0.188),
        ("mean_ulgy_qseq_int", -9.907),
        ("mean_ulgy_qseq_slp", 0.596),
        ("mean_ul_qseq_int", -2.790),
        ("mean_ul_qseq_slp", 1.725),
        ("mean_uh_qseq_int", -1.461),
        ("mean_uh_qseq_slp", -0.218),
        ("mean_uqt_int", 0.870),
        ("mean_uqt_slp", -0.319),
        ("mean_uqs_int", 0.602),
        ("mean_uqs_slp", -0.250),
        ("mean_udrop_int", -2.997),
        ("mean_udrop_slp", 0.431),
        ("mean_urej_int", -3.370),
        ("mean_urej_slp", 1.119),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", -0.127),
        ("std_ulgy_mseq_int", 0.078),
        ("std_ulgy_mseq_slp", 0.050),
        ("std_ul_mseq_int", 1.309),
        ("std_ul_mseq_slp", -0.908),
        ("std_uh_mseq_int", 1.005),
        ("std_uh_mseq_slp", 0.984),
        ("std_ulgm_qseq_int", 0.362),
        ("std_ulgm_qseq_slp", -0.065),
        ("std_ulgy_qseq_int", 0.055),
        ("std_ulgy_qseq_slp", -0.162),
        ("std_ul_qseq_int", 1.129),
        ("std_ul_qseq_slp", -0.999),
        ("std_uh_qseq_int", 0.050),
        ("std_uh_qseq_slp", -0.033),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.071),
        ("std_uqt_slp", 0.009),
        ("std_uqs_int", 0.032),
        ("std_uqs_slp", 0.046),
        ("std_udrop_int", 0.252),
        ("std_udrop_slp", -0.405),
        ("std_urej_int", 1.104),
        ("std_urej_slp", -0.999),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 10.225),
        ("frac_quench_cen_k_tpeak", 9.989),
        ("frac_quench_cen_x0_ylotpeak", 11.011),
        ("frac_quench_cen_x0_yhitpeak", 12.579),
        ("frac_quench_cen_ylo_ylotpeak", 0.222),
        ("frac_quench_cen_ylo_yhitpeak", 0.223),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.979),
        ("frac_quench_sat_x0_tpeak", 3.998),
        ("frac_quench_sat_k_tpeak", 4.400),
        ("frac_quench_sat_x0_ylotpeak", 11.870),
        ("frac_quench_sat_x0_yhitpeak", 11.456),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.203),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.887),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.001),
        ("delta_uqt_k", 0.673),
        ("delta_uqt_ylo", -0.857),
        ("delta_uqt_yhi", 0.131),
        ("delta_uqt_slope", -0.041),
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


DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS
)
