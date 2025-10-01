""""""

from collections import OrderedDict, namedtuple

from ..defaults_mgash import DiffstarPopParams, get_unbounded_diffstarpop_params
from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.167),
        ("mean_ulgm_mseq_ytp", 12.109),
        ("mean_ulgm_mseq_lo", 0.751),
        ("mean_ulgm_mseq_hi", 0.173),
        ("mean_ulgy_mseq_int", -9.812),
        ("mean_ulgy_mseq_slp", 0.625),
        ("mean_ul_mseq_int", -1.815),
        ("mean_ul_mseq_slp", 1.089),
        ("mean_uh_mseq_int", 0.746),
        ("mean_uh_mseq_slp", -0.726),
        ("mean_ulgm_qseq_xtp", 12.147),
        ("mean_ulgm_qseq_ytp", 12.145),
        ("mean_ulgm_qseq_lo", 0.940),
        ("mean_ulgm_qseq_hi", 0.046),
        ("mean_ulgy_qseq_int", -9.829),
        ("mean_ulgy_qseq_slp", 0.831),
        ("mean_ul_qseq_int", -1.954),
        ("mean_ul_qseq_slp", 0.784),
        ("mean_uh_qseq_int", 0.986),
        ("mean_uh_qseq_slp", -0.186),
        ("mean_uqt_int", 1.025),
        ("mean_uqt_slp", 0.013),
        ("mean_uqs_int", -0.226),
        ("mean_uqs_slp", 0.098),
        ("mean_udrop_int", -1.823),
        ("mean_udrop_slp", 0.071),
        ("mean_urej_int", -0.753),
        ("mean_urej_slp", -0.236),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.254),
        ("std_ulgm_mseq_slp", 0.154),
        ("std_ulgy_mseq_int", 0.269),
        ("std_ulgy_mseq_slp", 0.033),
        ("std_ul_mseq_int", 1.925),
        ("std_ul_mseq_slp", 0.544),
        ("std_uh_mseq_int", 1.463),
        ("std_uh_mseq_slp", 0.010),
        ("std_ulgm_qseq_int", 0.192),
        ("std_ulgm_qseq_slp", 0.127),
        ("std_ulgy_qseq_int", 0.287),
        ("std_ulgy_qseq_slp", -0.034),
        ("std_ul_qseq_int", 2.015),
        ("std_ul_qseq_slp", 0.181),
        ("std_uh_qseq_int", 1.041),
        ("std_uh_qseq_slp", 0.063),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.080),
        ("std_uqt_slp", 0.010),
        ("std_uqs_int", 0.613),
        ("std_uqs_slp", 0.221),
        ("std_udrop_int", 0.796),
        ("std_udrop_slp", -0.120),
        ("std_urej_int", 1.166),
        ("std_urej_slp", -0.021),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.238),
        ("frac_quench_cen_x0_yhitpeak", 11.780),
        ("frac_quench_cen_ylo_ylotpeak", 0.246),
        ("frac_quench_cen_ylo_yhitpeak", 0.027),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.238),
        ("frac_quench_sat_x0_yhitpeak", 11.780),
        ("frac_quench_sat_ylo_ylotpeak", 0.246),
        ("frac_quench_sat_ylo_yhitpeak", 0.027),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 2.846),
        ("delta_uqt_k", 0.515),
        ("delta_uqt_ylo", -0.484),
        ("delta_uqt_yhi", 0.036),
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


DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
