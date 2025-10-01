""""""

from collections import OrderedDict, namedtuple

from ..defaults_mgash import DiffstarPopParams, get_unbounded_diffstarpop_params
from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 13.561),
        ("mean_ulgm_mseq_ytp", 12.181),
        ("mean_ulgm_mseq_lo", 0.628),
        ("mean_ulgm_mseq_hi", -0.819),
        ("mean_ulgy_mseq_int", -9.663),
        ("mean_ulgy_mseq_slp", 0.105),
        ("mean_ul_mseq_int", -0.715),
        ("mean_ul_mseq_slp", 1.717),
        ("mean_uh_mseq_int", -0.392),
        ("mean_uh_mseq_slp", 0.043),
        ("mean_ulgm_qseq_xtp", 12.728),
        ("mean_ulgm_qseq_ytp", 11.931),
        ("mean_ulgm_qseq_lo", 0.769),
        ("mean_ulgm_qseq_hi", 0.123),
        ("mean_ulgy_qseq_int", -9.647),
        ("mean_ulgy_qseq_slp", 0.382),
        ("mean_ul_qseq_int", -0.378),
        ("mean_ul_qseq_slp", 1.047),
        ("mean_uh_qseq_int", -0.196),
        ("mean_uh_qseq_slp", 0.157),
        ("mean_uqt_int", 1.034),
        ("mean_uqt_slp", -0.096),
        ("mean_uqs_int", 0.040),
        ("mean_uqs_slp", -0.173),
        ("mean_udrop_int", -1.963),
        ("mean_udrop_slp", 0.746),
        ("mean_urej_int", -0.524),
        ("mean_urej_slp", 0.164),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.405),
        ("std_ulgm_mseq_slp", -0.094),
        ("std_ulgy_mseq_int", 0.229),
        ("std_ulgy_mseq_slp", -0.006),
        ("std_ul_mseq_int", 2.073),
        ("std_ul_mseq_slp", 0.374),
        ("std_uh_mseq_int", 1.176),
        ("std_uh_mseq_slp", -0.200),
        ("std_ulgm_qseq_int", 0.420),
        ("std_ulgm_qseq_slp", -0.019),
        ("std_ulgy_qseq_int", 0.227),
        ("std_ulgy_qseq_slp", -0.014),
        ("std_ul_qseq_int", 2.174),
        ("std_ul_qseq_slp", -0.137),
        ("std_uh_qseq_int", 0.761),
        ("std_uh_qseq_slp", -0.141),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.083),
        ("std_uqt_slp", 0.070),
        ("std_uqs_int", 0.900),
        ("std_uqs_slp", 0.002),
        ("std_udrop_int", 0.607),
        ("std_udrop_slp", 0.097),
        ("std_urej_int", 0.893),
        ("std_urej_slp", 0.260),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.708),
        ("frac_quench_cen_x0_yhitpeak", 13.900),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.413),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.708),
        ("frac_quench_sat_x0_yhitpeak", 13.900),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.413),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.977),
        ("delta_uqt_k", 1.334),
        ("delta_uqt_ylo", -0.367),
        ("delta_uqt_yhi", -0.008),
        ("delta_uqt_slope", -0.013),
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


DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    *SFH_PDF_QUENCH_PARAMS, *DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS
)
