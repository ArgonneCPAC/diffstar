from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.077),
        ("mean_ulgm_mseq_ytp", 11.198),
        ("mean_ulgm_mseq_lo", 0.324),
        ("mean_ulgm_mseq_hi", -0.612),
        ("mean_ulgy_mseq_xtp", 12.708),
        ("mean_ulgy_mseq_ytp", -9.579),
        ("mean_ulgy_mseq_lo", 0.753),
        ("mean_ulgy_mseq_hi", -3.358),
        ("mean_ul_mseq_int", -2.739),
        ("mean_ul_mseq_slp", 8.577),
        ("mean_uh_mseq_int", 0.164),
        ("mean_uh_mseq_slp", 1.255),
        ("mean_ulgm_qseq_xtp", 13.525),
        ("mean_ulgm_qseq_ytp", 12.100),
        ("mean_ulgm_qseq_lo", 0.420),
        ("mean_ulgm_qseq_hi", 0.423),
        ("mean_ulgy_qseq_xtp", 12.166),
        ("mean_ulgy_qseq_ytp", -9.571),
        ("mean_ulgy_qseq_lo", 0.702),
        ("mean_ulgy_qseq_hi", 0.005),
        ("mean_ul_qseq_int", -2.400),
        ("mean_ul_qseq_slp", 0.968),
        ("mean_uh_qseq_int", -0.915),
        ("mean_uh_qseq_slp", -0.459),
        ("mean_uqt_xtp", 12.555),
        ("mean_uqt_ytp", 1.035),
        ("mean_uqt_lo", -0.031),
        ("mean_uqt_hi", -0.589),
        ("mean_uqs_int", 1.451),
        ("mean_uqs_slp", 0.307),
        ("mean_udrop_int", -2.247),
        ("mean_udrop_slp", 1.486),
        ("mean_urej_int", -1.420),
        ("mean_urej_slp", 1.622),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.011),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", 0.009),
        ("std_ul_mseq_int", 1.725),
        ("std_ul_mseq_slp", -0.999),
        ("std_uh_mseq_int", 0.011),
        ("std_uh_mseq_slp", -0.981),
        ("std_ulgm_qseq_int", 0.095),
        ("std_ulgm_qseq_slp", 0.004),
        ("std_ulgy_qseq_int", 0.011),
        ("std_ulgy_qseq_slp", -0.122),
        ("std_ul_qseq_int", 2.224),
        ("std_ul_qseq_slp", -0.412),
        ("std_uh_qseq_int", 0.149),
        ("std_uh_qseq_slp", -0.996),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.077),
        ("std_uqt_slp", 0.045),
        ("std_uqs_int", 0.857),
        ("std_uqs_slp", 0.999),
        ("std_udrop_int", 0.510),
        ("std_udrop_slp", 0.334),
        ("std_urej_int", 0.020),
        ("std_urej_slp", 0.010),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.475),
        ("frac_quench_cen_k_tpeak", 2.293),
        ("frac_quench_cen_x0_ylotpeak", 11.675),
        ("frac_quench_cen_x0_yhitpeak", 12.484),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.001),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.995),
        ("frac_quench_sat_x0_tpeak", 8.624),
        ("frac_quench_sat_k_tpeak", 2.214),
        ("frac_quench_sat_x0_ylotpeak", 12.859),
        ("frac_quench_sat_x0_yhitpeak", 12.191),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.563),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.999),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.273),
        ("delta_uqt_k", 0.203),
        ("delta_uqt_ylo", -0.983),
        ("delta_uqt_yhi", 0.384),
        ("delta_uqt_slope", 0.025),
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
