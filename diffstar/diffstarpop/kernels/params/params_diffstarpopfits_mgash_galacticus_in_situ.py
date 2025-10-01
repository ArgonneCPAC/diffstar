""""""

from collections import OrderedDict, namedtuple

from ..defaults_mgash import DiffstarPopParams, get_unbounded_diffstarpop_params
from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.576),
        ("mean_ulgm_mseq_ytp", 11.226),
        ("mean_ulgm_mseq_lo", 0.252),
        ("mean_ulgm_mseq_hi", -2.900),
        ("mean_ulgy_mseq_int", -9.254),
        ("mean_ulgy_mseq_slp", 1.003),
        ("mean_ul_mseq_int", -2.997),
        ("mean_ul_mseq_slp", 12.403),
        ("mean_uh_mseq_int", -0.598),
        ("mean_uh_mseq_slp", 0.336),
        ("mean_ulgm_qseq_xtp", 12.167),
        ("mean_ulgm_qseq_ytp", 11.852),
        ("mean_ulgm_qseq_lo", 0.840),
        ("mean_ulgm_qseq_hi", 0.091),
        ("mean_ulgy_qseq_int", -9.671),
        ("mean_ulgy_qseq_slp", 0.312),
        ("mean_ul_qseq_int", -2.997),
        ("mean_ul_qseq_slp", 4.249),
        ("mean_uh_qseq_int", -1.484),
        ("mean_uh_qseq_slp", -0.251),
        ("mean_uqt_int", 0.927),
        ("mean_uqt_slp", -0.085),
        ("mean_uqs_int", 0.161),
        ("mean_uqs_slp", -0.798),
        ("mean_udrop_int", -1.809),
        ("mean_udrop_slp", 1.253),
        ("mean_urej_int", -4.271),
        ("mean_urej_slp", -0.664),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.001),
        ("std_ulgy_mseq_int", 0.046),
        ("std_ulgy_mseq_slp", 0.132),
        ("std_ul_mseq_int", 0.011),
        ("std_ul_mseq_slp", 0.005),
        ("std_uh_mseq_int", 0.011),
        ("std_uh_mseq_slp", -0.999),
        ("std_ulgm_qseq_int", 0.142),
        ("std_ulgm_qseq_slp", -0.021),
        ("std_ulgy_qseq_int", 0.011),
        ("std_ulgy_qseq_slp", -0.189),
        ("std_ul_qseq_int", 1.080),
        ("std_ul_qseq_slp", -0.999),
        ("std_uh_qseq_int", 0.011),
        ("std_uh_qseq_slp", 0.086),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.024),
        ("std_uqt_slp", 0.301),
        ("std_uqs_int", 0.711),
        ("std_uqs_slp", 0.458),
        ("std_udrop_int", 0.716),
        ("std_udrop_slp", 0.538),
        ("std_urej_int", 0.011),
        ("std_urej_slp", 0.002),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.501),
        ("frac_quench_cen_k_tpeak", 2.281),
        ("frac_quench_cen_x0_ylotpeak", 13.193),
        ("frac_quench_cen_x0_yhitpeak", 12.520),
        ("frac_quench_cen_ylo_ylotpeak", 0.588),
        ("frac_quench_cen_ylo_yhitpeak", 0.001),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.902),
        ("frac_quench_sat_x0_tpeak", 7.829),
        ("frac_quench_sat_k_tpeak", 9.990),
        ("frac_quench_sat_x0_ylotpeak", 11.011),
        ("frac_quench_sat_x0_yhitpeak", 12.406),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.510),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.999),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.864),
        ("delta_uqt_k", 0.290),
        ("delta_uqt_ylo", -0.781),
        ("delta_uqt_yhi", 0.295),
        ("delta_uqt_slope", 0.011),
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


DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS
)
