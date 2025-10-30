from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.842),
        ("mean_ulgm_mseq_ytp", 12.447),
        ("mean_ulgm_mseq_lo", 0.812),
        ("mean_ulgm_mseq_hi", 0.285),
        ("mean_ulgy_mseq_xtp", 11.674),
        ("mean_ulgy_mseq_ytp", -10.608),
        ("mean_ulgy_mseq_lo", 1.895),
        ("mean_ulgy_mseq_hi", 0.781),
        ("mean_ul_mseq_int", -2.997),
        ("mean_ul_mseq_slp", 0.001),
        ("mean_uh_mseq_int", -4.995),
        ("mean_uh_mseq_slp", -2.820),
        ("mean_ulgm_qseq_xtp", 11.011),
        ("mean_ulgm_qseq_ytp", 12.142),
        ("mean_ulgm_qseq_lo", 4.995),
        ("mean_ulgm_qseq_hi", 0.010),
        ("mean_ulgy_qseq_xtp", 12.836),
        ("mean_ulgy_qseq_ytp", -9.690),
        ("mean_ulgy_qseq_lo", 0.602),
        ("mean_ulgy_qseq_hi", 0.842),
        ("mean_ul_qseq_int", -2.997),
        ("mean_ul_qseq_slp", 0.001),
        ("mean_uh_qseq_int", -2.241),
        ("mean_uh_qseq_slp", 0.606),
        ("mean_uqt_xtp", 13.533),
        ("mean_uqt_ytp", 1.127),
        ("mean_uqt_lo", -0.011),
        ("mean_uqt_hi", -0.238),
        ("mean_uqs_int", 1.028),
        ("mean_uqs_slp", 2.132),
        ("mean_udrop_int", -2.177),
        ("mean_udrop_slp", 0.308),
        ("mean_urej_int", -4.516),
        ("mean_urej_slp", -2.300),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.074),
        ("std_ulgm_mseq_slp", -0.106),
        ("std_ulgy_mseq_int", 0.011),
        ("std_ulgy_mseq_slp", 0.001),
        ("std_ul_mseq_int", 0.014),
        ("std_ul_mseq_slp", 0.009),
        ("std_uh_mseq_int", 0.011),
        ("std_uh_mseq_slp", -0.999),
        ("std_ulgm_qseq_int", 0.243),
        ("std_ulgm_qseq_slp", -0.162),
        ("std_ulgy_qseq_int", 0.091),
        ("std_ulgy_qseq_slp", -0.229),
        ("std_ul_qseq_int", 0.013),
        ("std_ul_qseq_slp", 0.007),
        ("std_uh_qseq_int", 0.702),
        ("std_uh_qseq_slp", -0.274),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.029),
        ("std_uqt_slp", 0.041),
        ("std_uqs_int", 0.011),
        ("std_uqs_slp", 0.001),
        ("std_udrop_int", 0.599),
        ("std_udrop_slp", -0.868),
        ("std_urej_int", 0.974),
        ("std_urej_slp", -0.050),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 11.207),
        ("frac_quench_cen_k_tpeak", 9.998),
        ("frac_quench_cen_x0_ylotpeak", 11.001),
        ("frac_quench_cen_x0_yhitpeak", 12.606),
        ("frac_quench_cen_ylo_ylotpeak", 0.999),
        ("frac_quench_cen_ylo_yhitpeak", 0.001),
        ("frac_quench_cen_k", 1.954),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 9.372),
        ("frac_quench_sat_k_tpeak", 9.995),
        ("frac_quench_sat_x0_ylotpeak", 11.810),
        ("frac_quench_sat_x0_yhitpeak", 11.547),
        ("frac_quench_sat_ylo_ylotpeak", 0.524),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.995),
        ("frac_quench_sat_yhi", 0.797),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 2.173),
        ("delta_uqt_k", 0.110),
        ("delta_uqt_ylo", -0.713),
        ("delta_uqt_yhi", 0.217),
        ("delta_uqt_slope", -0.023),
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

DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
