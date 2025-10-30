from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.029),
        ("mean_ulgm_mseq_ytp", 11.300),
        ("mean_ulgm_mseq_lo", 2.812),
        ("mean_ulgm_mseq_hi", 0.400),
        ("mean_ulgy_mseq_xtp", 12.344),
        ("mean_ulgy_mseq_ytp", -9.296),
        ("mean_ulgy_mseq_lo", 0.902),
        ("mean_ulgy_mseq_hi", 0.303),
        ("mean_ul_mseq_int", -0.639),
        ("mean_ul_mseq_slp", 0.988),
        ("mean_uh_mseq_int", -0.391),
        ("mean_uh_mseq_slp", 0.161),
        ("mean_ulgm_qseq_xtp", 13.078),
        ("mean_ulgm_qseq_ytp", 11.903),
        ("mean_ulgm_qseq_lo", 0.794),
        ("mean_ulgm_qseq_hi", 0.230),
        ("mean_ulgy_qseq_xtp", 12.112),
        ("mean_ulgy_qseq_ytp", -9.421),
        ("mean_ulgy_qseq_lo", 0.725),
        ("mean_ulgy_qseq_hi", 0.356),
        ("mean_ul_qseq_int", -0.714),
        ("mean_ul_qseq_slp", 0.616),
        ("mean_uh_qseq_int", -0.582),
        ("mean_uh_qseq_slp", -0.115),
        ("mean_uqt_xtp", 12.922),
        ("mean_uqt_ytp", 1.047),
        ("mean_uqt_lo", -0.100),
        ("mean_uqt_hi", -0.100),
        ("mean_uqs_int", -0.298),
        ("mean_uqs_slp", -0.411),
        ("mean_udrop_int", -1.953),
        ("mean_udrop_slp", 0.586),
        ("mean_urej_int", -0.853),
        ("mean_urej_slp", 0.329),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.552),
        ("std_ulgm_mseq_slp", -0.433),
        ("std_ulgy_mseq_int", 0.298),
        ("std_ulgy_mseq_slp", 0.046),
        ("std_ul_mseq_int", 2.469),
        ("std_ul_mseq_slp", 0.677),
        ("std_uh_mseq_int", 0.705),
        ("std_uh_mseq_slp", -0.287),
        ("std_ulgm_qseq_int", 0.492),
        ("std_ulgm_qseq_slp", -0.293),
        ("std_ulgy_qseq_int", 0.264),
        ("std_ulgy_qseq_slp", 0.063),
        ("std_ul_qseq_int", 2.263),
        ("std_ul_qseq_slp", -0.165),
        ("std_uh_qseq_int", 0.528),
        ("std_uh_qseq_slp", -0.067),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.075),
        ("std_uqt_slp", -0.008),
        ("std_uqs_int", 0.900),
        ("std_uqs_slp", 0.900),
        ("std_udrop_int", 0.600),
        ("std_udrop_slp", 0.020),
        ("std_urej_int", 0.979),
        ("std_urej_slp", -0.205),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.750),
        ("frac_quench_cen_x0_yhitpeak", 12.965),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.625),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.750),
        ("frac_quench_sat_x0_yhitpeak", 12.965),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.625),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.001),
        ("delta_uqt_k", 0.714),
        ("delta_uqt_ylo", -0.611),
        ("delta_uqt_yhi", 0.002),
        ("delta_uqt_slope", -0.024),
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

DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
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
