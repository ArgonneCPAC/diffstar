from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 11.986),
        ("mean_ulgm_mseq_ytp", 11.189),
        ("mean_ulgm_mseq_lo", 0.370),
        ("mean_ulgm_mseq_hi", 0.147),
        ("mean_ulgy_mseq_xtp", 12.677),
        ("mean_ulgy_mseq_ytp", -9.236),
        ("mean_ulgy_mseq_lo", 0.934),
        ("mean_ulgy_mseq_hi", -2.625),
        ("mean_ul_mseq_int", -2.997),
        ("mean_ul_mseq_slp", 0.001),
        ("mean_uh_mseq_int", -0.596),
        ("mean_uh_mseq_slp", 0.708),
        ("mean_ulgm_qseq_xtp", 13.105),
        ("mean_ulgm_qseq_ytp", 11.937),
        ("mean_ulgm_qseq_lo", 0.504),
        ("mean_ulgm_qseq_hi", 0.616),
        ("mean_ulgy_qseq_xtp", 11.979),
        ("mean_ulgy_qseq_ytp", -9.432),
        ("mean_ulgy_qseq_lo", 0.995),
        ("mean_ulgy_qseq_hi", 0.310),
        ("mean_ul_qseq_int", -2.997),
        ("mean_ul_qseq_slp", -1.731),
        ("mean_uh_qseq_int", -1.134),
        ("mean_uh_qseq_slp", -0.500),
        ("mean_uqt_xtp", 12.400),
        ("mean_uqt_ytp", 1.104),
        ("mean_uqt_lo", -0.053),
        ("mean_uqt_hi", -0.209),
        ("mean_uqs_int", 1.676),
        ("mean_uqs_slp", 0.509),
        ("mean_udrop_int", -2.189),
        ("mean_udrop_slp", 1.064),
        ("mean_urej_int", -3.147),
        ("mean_urej_slp", 0.202),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", -0.081),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", 0.234),
        ("std_ul_mseq_int", 0.013),
        ("std_ul_mseq_slp", 0.006),
        ("std_uh_mseq_int", 0.012),
        ("std_uh_mseq_slp", 0.005),
        ("std_ulgm_qseq_int", 0.037),
        ("std_ulgm_qseq_slp", 0.030),
        ("std_ulgy_qseq_int", 0.025),
        ("std_ulgy_qseq_slp", 0.146),
        ("std_ul_qseq_int", 0.011),
        ("std_ul_qseq_slp", -0.999),
        ("std_uh_qseq_int", 0.301),
        ("std_uh_qseq_slp", -0.083),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.114),
        ("std_uqt_slp", 0.057),
        ("std_uqs_int", 0.999),
        ("std_uqs_slp", 0.930),
        ("std_udrop_int", 0.512),
        ("std_udrop_slp", 0.280),
        ("std_urej_int", 0.025),
        ("std_urej_slp", 0.008),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.475),
        ("frac_quench_cen_k_tpeak", 2.287),
        ("frac_quench_cen_x0_ylotpeak", 11.742),
        ("frac_quench_cen_x0_yhitpeak", 12.605),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.007),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.997),
        ("frac_quench_sat_x0_tpeak", 7.617),
        ("frac_quench_sat_k_tpeak", 9.998),
        ("frac_quench_sat_x0_ylotpeak", 13.994),
        ("frac_quench_sat_x0_yhitpeak", 12.583),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.665),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.999),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 2.423),
        ("delta_uqt_k", 0.285),
        ("delta_uqt_ylo", -0.758),
        ("delta_uqt_yhi", 0.227),
        ("delta_uqt_slope", 0.049),
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

DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
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
