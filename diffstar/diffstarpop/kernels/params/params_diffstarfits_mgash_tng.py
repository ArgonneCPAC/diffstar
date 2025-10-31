from collections import OrderedDict, namedtuple

from ..defaults_mgash import (
    DiffstarPopParams,
    get_unbounded_diffstarpop_params,
)

from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 11.300),
        ("mean_ulgm_mseq_ytp", 11.300),
        ("mean_ulgm_mseq_lo", 1.296),
        ("mean_ulgm_mseq_hi", 0.336),
        ("mean_ulgy_mseq_xtp", 13.509),
        ("mean_ulgy_mseq_ytp", -9.297),
        ("mean_ulgy_mseq_lo", 0.543),
        ("mean_ulgy_mseq_hi", 0.247),
        ("mean_ul_mseq_int", -0.358),
        ("mean_ul_mseq_slp", 1.187),
        ("mean_uh_mseq_int", -0.807),
        ("mean_uh_mseq_slp", 0.066),
        ("mean_ulgm_qseq_xtp", 12.622),
        ("mean_ulgm_qseq_ytp", 11.767),
        ("mean_ulgm_qseq_lo", 0.289),
        ("mean_ulgm_qseq_hi", 0.297),
        ("mean_ulgy_qseq_xtp", 12.334),
        ("mean_ulgy_qseq_ytp", -9.655),
        ("mean_ulgy_qseq_lo", 0.572),
        ("mean_ulgy_qseq_hi", 0.388),
        ("mean_ul_qseq_int", -0.354),
        ("mean_ul_qseq_slp", 1.694),
        ("mean_uh_qseq_int", -0.924),
        ("mean_uh_qseq_slp", -0.869),
        ("mean_uqt_xtp", 13.226),
        ("mean_uqt_ytp", 0.929),
        ("mean_uqt_lo", -0.100),
        ("mean_uqt_hi", -0.100),
        ("mean_uqs_int", 0.144),
        ("mean_uqs_slp", -0.081),
        ("mean_udrop_int", -2.026),
        ("mean_udrop_slp", 0.543),
        ("mean_urej_int", -0.932),
        ("mean_urej_slp", 0.409),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.379),
        ("std_ulgm_mseq_slp", 0.025),
        ("std_ulgy_mseq_int", 0.283),
        ("std_ulgy_mseq_slp", 0.007),
        ("std_ul_mseq_int", 1.665),
        ("std_ul_mseq_slp", 0.337),
        ("std_uh_mseq_int", 1.588),
        ("std_uh_mseq_slp", -0.254),
        ("std_ulgm_qseq_int", 0.362),
        ("std_ulgm_qseq_slp", 0.001),
        ("std_ulgy_qseq_int", 0.274),
        ("std_ulgy_qseq_slp", -0.008),
        ("std_ul_qseq_int", 1.766),
        ("std_ul_qseq_slp", -0.476),
        ("std_uh_qseq_int", 1.306),
        ("std_uh_qseq_slp", -0.208),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.131),
        ("std_uqt_slp", 0.071),
        ("std_uqs_int", 0.590),
        ("std_uqs_slp", 0.184),
        ("std_udrop_int", 0.784),
        ("std_udrop_slp", -0.065),
        ("std_urej_int", 1.221),
        ("std_urej_slp", -0.111),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 12.813),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.196),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.100),
        ("frac_quench_sat_x0_yhitpeak", 12.813),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.196),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 3.309),
        ("delta_uqt_k", 4.719),
        ("delta_uqt_ylo", -0.308),
        ("delta_uqt_yhi", 0.029),
        ("delta_uqt_slope", 0.009),
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
