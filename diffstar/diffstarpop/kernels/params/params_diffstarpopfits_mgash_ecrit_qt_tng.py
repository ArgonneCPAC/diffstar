from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 11.934),
        ("mean_ulgm_mseq_ytp", 11.392),
        ("mean_ulgm_mseq_lo", 0.654),
        ("mean_ulgm_mseq_hi", 0.409),
        ("mean_ulgy_mseq_xtp", 12.716),
        ("mean_ulgy_mseq_ytp", -9.372),
        ("mean_ulgy_mseq_lo", 0.691),
        ("mean_ulgy_mseq_hi", 0.999),
        ("mean_ul_mseq_int", 0.350),
        ("mean_ul_mseq_slp", 2.462),
        ("mean_uh_mseq_int", -2.144),
        ("mean_uh_mseq_slp", -0.700),
        ("mean_ulgm_qseq_xtp", 13.590),
        ("mean_ulgm_qseq_ytp", 11.911),
        ("mean_ulgm_qseq_lo", 0.218),
        ("mean_ulgm_qseq_hi", 0.294),
        ("mean_ulgy_qseq_xtp", 12.038),
        ("mean_ulgy_qseq_ytp", -9.768),
        ("mean_ulgy_qseq_lo", 1.350),
        ("mean_ulgy_qseq_hi", 0.597),
        ("mean_ul_qseq_int", -1.536),
        ("mean_ul_qseq_slp", -0.132),
        ("mean_uh_qseq_int", -1.129),
        ("mean_uh_qseq_slp", -0.216),
        ("mean_uqt_xtp", 13.551),
        ("mean_uqt_ytp", 0.698),
        ("mean_uqt_lo", -0.332),
        ("mean_uqt_hi", -0.018),
        ("mean_uqs_int", 1.324),
        ("mean_uqs_slp", 0.301),
        ("mean_udrop_int", -2.997),
        ("mean_udrop_slp", 1.093),
        ("mean_urej_int", -9.199),
        ("mean_urej_slp", -0.854),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.001),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", -0.184),
        ("std_ul_mseq_int", 0.085),
        ("std_ul_mseq_slp", 0.994),
        ("std_uh_mseq_int", 0.103),
        ("std_uh_mseq_slp", -0.999),
        ("std_ulgm_qseq_int", 0.011),
        ("std_ulgm_qseq_slp", -0.175),
        ("std_ulgy_qseq_int", 0.011),
        ("std_ulgy_qseq_slp", -0.122),
        ("std_ul_qseq_int", 0.014),
        ("std_ul_qseq_slp", -0.002),
        ("std_uh_qseq_int", 0.575),
        ("std_uh_qseq_slp", -0.469),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.073),
        ("std_uqt_slp", -0.009),
        ("std_uqs_int", 0.042),
        ("std_uqs_slp", -0.035),
        ("std_udrop_int", 0.011),
        ("std_udrop_slp", 0.753),
        ("std_urej_int", 0.127),
        ("std_urej_slp", -0.063),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 13.615),
        ("frac_quench_cen_k_tpeak", 9.240),
        ("frac_quench_cen_x0_ylotpeak", 13.966),
        ("frac_quench_cen_x0_yhitpeak", 11.541),
        ("frac_quench_cen_ylo_ylotpeak", 0.041),
        ("frac_quench_cen_ylo_yhitpeak", 0.737),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 11.905),
        ("frac_quench_sat_k_tpeak", 4.158),
        ("frac_quench_sat_x0_ylotpeak", 12.469),
        ("frac_quench_sat_x0_yhitpeak", 12.456),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.999),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 2.532),
        ("delta_uqt_k", 0.454),
        ("delta_uqt_ylo", -0.977),
        ("delta_uqt_yhi", -0.002),
        ("delta_uqt_slope", -0.030),
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

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS
)
