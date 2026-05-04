from collections import OrderedDict, namedtuple

from ..defaults_mgash import DiffstarPopParams, get_unbounded_diffstarpop_params

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.449),
        ("mean_ulgm_mseq_ytp", 11.897),
        ("mean_ulgm_mseq_lo", 0.992),
        ("mean_ulgm_mseq_hi", 0.039),
        ("mean_ulgy_mseq_xtp", 13.459),
        ("mean_ulgy_mseq_ytp", -8.471),
        ("mean_ulgy_mseq_lo", 1.249),
        ("mean_ulgy_mseq_hi", 0.465),
        ("mean_ul_mseq_int", -0.714),
        ("mean_ul_mseq_slp", 3.270),
        ("mean_uh_mseq_int", -4.088),
        ("mean_uh_mseq_slp", -4.347),
        ("mean_ulgm_qseq_xtp", 11.073),
        ("mean_ulgm_qseq_ytp", 11.002),
        ("mean_ulgm_qseq_lo", 3.635),
        ("mean_ulgm_qseq_hi", 0.551),
        ("mean_ulgy_qseq_xtp", 12.971),
        ("mean_ulgy_qseq_ytp", -9.536),
        ("mean_ulgy_qseq_lo", 0.638),
        ("mean_ulgy_qseq_hi", 0.266),
        ("mean_ul_qseq_int", -2.147),
        ("mean_ul_qseq_slp", -1.249),
        ("mean_uh_qseq_int", -1.392),
        ("mean_uh_qseq_slp", -0.790),
        ("mean_uqt_xtp", 12.107),
        ("mean_uqt_ytp", 1.084),
        ("mean_uqt_lo", -0.057),
        ("mean_uqt_hi", -0.519),
        ("mean_uqs_int", -0.019),
        ("mean_uqs_slp", -0.231),
        ("mean_udrop_int", -2.999),
        ("mean_udrop_slp", 2.113),
        ("mean_urej_int", -8.730),
        ("mean_urej_slp", -0.280),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.011),
        ("std_ulgm_mseq_slp", 0.049),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", -0.123),
        ("std_ul_mseq_int", 0.768),
        ("std_ul_mseq_slp", 0.997),
        ("std_uh_mseq_int", 0.457),
        ("std_uh_mseq_slp", -0.999),
        ("std_ulgm_qseq_int", 0.108),
        ("std_ulgm_qseq_slp", -0.149),
        ("std_ulgy_qseq_int", 0.071),
        ("std_ulgy_qseq_slp", -0.077),
        ("std_ul_qseq_int", 0.392),
        ("std_ul_qseq_slp", -0.681),
        ("std_uh_qseq_int", 0.013),
        ("std_uh_qseq_slp", 0.432),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.028),
        ("std_uqt_slp", -0.013),
        ("std_uqs_int", 0.075),
        ("std_uqs_slp", -0.038),
        ("std_udrop_int", 0.463),
        ("std_udrop_slp", -0.824),
        ("std_urej_int", 0.131),
        ("std_urej_slp", 0.013),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 13.080),
        ("frac_quench_cen_k_tpeak", 9.936),
        ("frac_quench_cen_x0_ylotpeak", 12.066),
        ("frac_quench_cen_x0_yhitpeak", 12.401),
        ("frac_quench_cen_ylo_ylotpeak", 0.999),
        ("frac_quench_cen_ylo_yhitpeak", 0.001),
        ("frac_quench_cen_k", 4.999),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 9.388),
        ("frac_quench_sat_k_tpeak", 9.848),
        ("frac_quench_sat_x0_ylotpeak", 13.025),
        ("frac_quench_sat_x0_yhitpeak", 12.397),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.999),
        ("frac_quench_sat_yhi", 0.848),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 4.191),
        ("delta_uqt_k", 0.577),
        ("delta_uqt_ylo", -0.496),
        ("delta_uqt_yhi", 0.094),
        ("delta_uqt_slope", -0.048),
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
