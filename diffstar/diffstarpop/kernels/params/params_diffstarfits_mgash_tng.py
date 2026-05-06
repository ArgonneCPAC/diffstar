from collections import OrderedDict, namedtuple

from ..defaults_mgash import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 11.738),
        ("mean_ulgm_mseq_ytp", 11.530),
        ("mean_ulgm_mseq_lo", 1.144),
        ("mean_ulgm_mseq_hi", 0.321),
        ("mean_ulgy_mseq_xtp", 13.700),
        ("mean_ulgy_mseq_ytp", -9.450),
        ("mean_ulgy_mseq_lo", 0.514),
        ("mean_ulgy_mseq_hi", -0.689),
        ("mean_ul_mseq_int", -0.413),
        ("mean_ul_mseq_slp", 0.872),
        ("mean_uh_mseq_int", -1.221),
        ("mean_uh_mseq_slp", -0.443),
        ("mean_ulgm_qseq_xtp", 11.300),
        ("mean_ulgm_qseq_ytp", 11.362),
        ("mean_ulgm_qseq_lo", 1.505),
        ("mean_ulgm_qseq_hi", 0.340),
        ("mean_ulgy_qseq_xtp", 12.335),
        ("mean_ulgy_qseq_ytp", -9.903),
        ("mean_ulgy_qseq_lo", 0.533),
        ("mean_ulgy_qseq_hi", 0.400),
        ("mean_ul_qseq_int", -0.629),
        ("mean_ul_qseq_slp", 1.597),
        ("mean_uh_qseq_int", -0.877),
        ("mean_uh_qseq_slp", -1.015),
        ("mean_uqt_xtp", 12.141),
        ("mean_uqt_ytp", 1.035),
        ("mean_uqt_lo", -0.100),
        ("mean_uqt_hi", -0.100),
        ("mean_uqs_int", 0.136),
        ("mean_uqs_slp", -0.008),
        ("mean_udrop_int", -2.077),
        ("mean_udrop_slp", 0.489),
        ("mean_urej_int", -1.012),
        ("mean_urej_slp", 0.420),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.383),
        ("std_ulgm_mseq_slp", 0.053),
        ("std_ulgy_mseq_int", 0.276),
        ("std_ulgy_mseq_slp", -0.005),
        ("std_ul_mseq_int", 1.677),
        ("std_ul_mseq_slp", 0.017),
        ("std_uh_mseq_int", 1.673),
        ("std_uh_mseq_slp", -0.196),
        ("std_ulgm_qseq_int", 0.378),
        ("std_ulgm_qseq_slp", -0.091),
        ("std_ulgy_qseq_int", 0.281),
        ("std_ulgy_qseq_slp", -0.024),
        ("std_ul_qseq_int", 1.832),
        ("std_ul_qseq_slp", -0.537),
        ("std_uh_qseq_int", 1.448),
        ("std_uh_qseq_slp", -0.102),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.129),
        ("std_uqt_slp", 0.069),
        ("std_uqs_int", 0.560),
        ("std_uqs_slp", 0.175),
        ("std_udrop_int", 0.765),
        ("std_udrop_slp", 0.018),
        ("std_urej_int", 1.222),
        ("std_urej_slp", -0.118),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 12.914),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.185),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.100),
        ("frac_quench_sat_x0_yhitpeak", 12.914),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.185),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 3.364),
        ("delta_uqt_k", 4.982),
        ("delta_uqt_ylo", -0.295),
        ("delta_uqt_yhi", 0.026),
        ("delta_uqt_slope", -0.003),
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

DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS
)
