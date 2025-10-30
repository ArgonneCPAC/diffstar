from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.900),
        ("mean_ulgm_mseq_ytp", 12.526),
        ("mean_ulgm_mseq_lo", 0.745),
        ("mean_ulgm_mseq_hi", -0.061),
        ("mean_ulgy_mseq_xtp", 11.885),
        ("mean_ulgy_mseq_ytp", -10.536),
        ("mean_ulgy_mseq_lo", 1.644),
        ("mean_ulgy_mseq_hi", 0.461),
        ("mean_ul_mseq_int", -2.806),
        ("mean_ul_mseq_slp", -0.321),
        ("mean_uh_mseq_int", -4.995),
        ("mean_uh_mseq_slp", 1.479),
        ("mean_ulgm_qseq_xtp", 13.349),
        ("mean_ulgm_qseq_ytp", 12.240),
        ("mean_ulgm_qseq_lo", 0.345),
        ("mean_ulgm_qseq_hi", -0.122),
        ("mean_ulgy_qseq_xtp", 11.938),
        ("mean_ulgy_qseq_ytp", -9.896),
        ("mean_ulgy_qseq_lo", 0.922),
        ("mean_ulgy_qseq_hi", 0.450),
        ("mean_ul_qseq_int", -0.597),
        ("mean_ul_qseq_slp", 0.255),
        ("mean_uh_qseq_int", -1.440),
        ("mean_uh_qseq_slp", -0.799),
        ("mean_uqt_xtp", 11.305),
        ("mean_uqt_ytp", 1.181),
        ("mean_uqt_lo", -0.001),
        ("mean_uqt_hi", -0.445),
        ("mean_uqs_int", 0.665),
        ("mean_uqs_slp", 0.073),
        ("mean_udrop_int", -2.997),
        ("mean_udrop_slp", 0.960),
        ("mean_urej_int", -4.668),
        ("mean_urej_slp", 2.971),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.063),
        ("std_ulgm_mseq_slp", -0.058),
        ("std_ulgy_mseq_int", 0.057),
        ("std_ulgy_mseq_slp", 0.246),
        ("std_ul_mseq_int", 0.871),
        ("std_ul_mseq_slp", -0.999),
        ("std_uh_mseq_int", 0.011),
        ("std_uh_mseq_slp", 0.014),
        ("std_ulgm_qseq_int", 0.440),
        ("std_ulgm_qseq_slp", -0.309),
        ("std_ulgy_qseq_int", 0.045),
        ("std_ulgy_qseq_slp", 0.119),
        ("std_ul_qseq_int", 0.762),
        ("std_ul_qseq_slp", 0.636),
        ("std_uh_qseq_int", 0.011),
        ("std_uh_qseq_slp", 0.001),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.054),
        ("std_uqt_slp", 0.021),
        ("std_uqs_int", 0.077),
        ("std_uqs_slp", 0.074),
        ("std_udrop_int", 0.288),
        ("std_udrop_slp", -0.999),
        ("std_urej_int", 0.965),
        ("std_urej_slp", -0.998),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 9.731),
        ("frac_quench_cen_k_tpeak", 9.989),
        ("frac_quench_cen_x0_ylotpeak", 13.998),
        ("frac_quench_cen_x0_yhitpeak", 13.654),
        ("frac_quench_cen_ylo_ylotpeak", 0.999),
        ("frac_quench_cen_ylo_yhitpeak", 0.302),
        ("frac_quench_cen_k", 2.169),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 9.464),
        ("frac_quench_sat_k_tpeak", 0.587),
        ("frac_quench_sat_x0_ylotpeak", 11.097),
        ("frac_quench_sat_x0_yhitpeak", 13.986),
        ("frac_quench_sat_ylo_ylotpeak", 0.501),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.999),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.001),
        ("delta_uqt_k", 0.578),
        ("delta_uqt_ylo", -0.569),
        ("delta_uqt_yhi", 0.309),
        ("delta_uqt_slope", 0.098),
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

DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS
)
