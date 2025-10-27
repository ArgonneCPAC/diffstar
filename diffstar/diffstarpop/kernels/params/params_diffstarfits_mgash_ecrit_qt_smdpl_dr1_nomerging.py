from collections import OrderedDict, namedtuple

from ..defaults_mgash_ecrit_qt import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.133),
        ("mean_ulgm_mseq_ytp", 12.097),
        ("mean_ulgm_mseq_lo", 0.793),
        ("mean_ulgm_mseq_hi", 0.104),
        ("mean_ulgy_mseq_xtp", 13.085),
        ("mean_ulgy_mseq_ytp", -9.718),
        ("mean_ulgy_mseq_lo", 0.638),
        ("mean_ulgy_mseq_hi", 0.144),
        ("mean_ul_mseq_int", -0.804),
        ("mean_ul_mseq_slp", 0.743),
        ("mean_uh_mseq_int", -1.504),
        ("mean_uh_mseq_slp", -2.294),
        ("mean_ulgm_qseq_xtp", 12.264),
        ("mean_ulgm_qseq_ytp", 12.232),
        ("mean_ulgm_qseq_lo", 0.816),
        ("mean_ulgm_qseq_hi", 0.075),
        ("mean_ulgy_qseq_xtp", 12.567),
        ("mean_ulgy_qseq_ytp", -9.840),
        ("mean_ulgy_qseq_lo", 0.579),
        ("mean_ulgy_qseq_hi", 0.329),
        ("mean_ul_qseq_int", -0.847),
        ("mean_ul_qseq_slp", 0.753),
        ("mean_uh_qseq_int", 1.142),
        ("mean_uh_qseq_slp", 0.581),
        ("mean_uqt_xtp", 12.862),
        ("mean_uqt_ytp", 0.902),
        ("mean_uqt_lo", -0.124),
        ("mean_uqt_hi", -0.267),
        ("mean_uqs_int", -0.109),
        ("mean_uqs_slp", 0.391),
        ("mean_udrop_int", -2.140),
        ("mean_udrop_slp", 0.229),
        ("mean_urej_int", -1.217),
        ("mean_urej_slp", -0.024),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.288),
        ("std_ulgm_mseq_slp", 0.145),
        ("std_ulgy_mseq_int", 0.289),
        ("std_ulgy_mseq_slp", 0.034),
        ("std_ul_mseq_int", 1.284),
        ("std_ul_mseq_slp", 0.415),
        ("std_uh_mseq_int", 2.477),
        ("std_uh_mseq_slp", -0.900),
        ("std_ulgm_qseq_int", 0.238),
        ("std_ulgm_qseq_slp", 0.125),
        ("std_ulgy_qseq_int", 0.257),
        ("std_ulgy_qseq_slp", -0.220),
        ("std_ul_qseq_int", 1.661),
        ("std_ul_qseq_slp", -0.455),
        ("std_uh_qseq_int", 1.484),
        ("std_uh_qseq_slp", 0.218),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.121),
        ("std_uqt_slp", 0.053),
        ("std_uqs_int", 0.565),
        ("std_uqs_slp", -0.027),
        ("std_udrop_int", 0.745),
        ("std_udrop_slp", 0.168),
        ("std_urej_int", 1.359),
        ("std_urej_slp", 0.066),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.349),
        ("frac_quench_cen_x0_yhitpeak", 11.862),
        ("frac_quench_cen_ylo_ylotpeak", 0.038),
        ("frac_quench_cen_ylo_yhitpeak", 0.010),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.349),
        ("frac_quench_sat_x0_yhitpeak", 11.862),
        ("frac_quench_sat_ylo_ylotpeak", 0.038),
        ("frac_quench_sat_ylo_yhitpeak", 0.010),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.548),
        ("delta_uqt_k", 0.493),
        ("delta_uqt_ylo", -0.494),
        ("delta_uqt_yhi", 0.023),
        ("delta_uqt_slope", -0.049),
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

DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS
)
