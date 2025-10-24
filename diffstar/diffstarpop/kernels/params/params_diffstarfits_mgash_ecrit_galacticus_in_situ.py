from collections import OrderedDict, namedtuple

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash_ecrit import DiffstarPopParams, get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 13.189),
        ("mean_ulgm_mseq_ytp", 12.079),
        ("mean_ulgm_mseq_lo", 0.783),
        ("mean_ulgm_mseq_hi", -0.001),
        ("mean_ulgy_mseq_xtp", 11.995),
        ("mean_ulgy_mseq_ytp", -9.781),
        ("mean_ulgy_mseq_lo", 0.927),
        ("mean_ulgy_mseq_hi", 0.091),
        ("mean_ul_mseq_int", -0.715),
        ("mean_ul_mseq_slp", 1.717),
        ("mean_uh_mseq_int", -0.253),
        ("mean_uh_mseq_slp", 0.364),
        ("mean_ulgm_qseq_xtp", 13.473),
        ("mean_ulgm_qseq_ytp", 12.089),
        ("mean_ulgm_qseq_lo", 0.404),
        ("mean_ulgm_qseq_hi", -0.228),
        ("mean_ulgy_qseq_xtp", 12.303),
        ("mean_ulgy_qseq_ytp", -9.608),
        ("mean_ulgy_qseq_lo", 0.446),
        ("mean_ulgy_qseq_hi", 0.017),
        ("mean_ul_qseq_int", -0.388),
        ("mean_ul_qseq_slp", 0.954),
        ("mean_uh_qseq_int", -0.153),
        ("mean_uh_qseq_slp", 0.284),
        ("mean_uqt_int", 0.994),
        ("mean_uqt_slp", -0.042),
        ("mean_uqs_int", 0.057),
        ("mean_uqs_slp", -0.196),
        ("mean_udrop_int", -1.917),
        ("mean_udrop_slp", 0.684),
        ("mean_urej_int", -0.749),
        ("mean_urej_slp", 0.472),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.527),
        ("std_ulgm_mseq_slp", -0.248),
        ("std_ulgy_mseq_int", 0.247),
        ("std_ulgy_mseq_slp", -0.031),
        ("std_ul_mseq_int", 2.156),
        ("std_ul_mseq_slp", 0.260),
        ("std_uh_mseq_int", 1.112),
        ("std_uh_mseq_slp", -0.113),
        ("std_ulgm_qseq_int", 0.434),
        ("std_ulgm_qseq_slp", -0.097),
        ("std_ulgy_qseq_int", 0.229),
        ("std_ulgy_qseq_slp", -0.018),
        ("std_ul_qseq_int", 2.179),
        ("std_ul_qseq_slp", -0.124),
        ("std_uh_qseq_int", 0.770),
        ("std_uh_qseq_slp", -0.177),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.110),
        ("std_uqt_slp", 0.032),
        ("std_uqs_int", 0.945),
        ("std_uqs_slp", 0.120),
        ("std_udrop_int", 0.639),
        ("std_udrop_slp", 0.053),
        ("std_urej_int", 1.118),
        ("std_urej_slp", -0.044),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.692),
        ("frac_quench_cen_x0_yhitpeak", 12.963),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.525),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.692),
        ("frac_quench_sat_x0_yhitpeak", 12.963),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.525),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.659),
        ("delta_uqt_k", 0.939),
        ("delta_uqt_ylo", -0.466),
        ("delta_uqt_yhi", 0.000),
        ("delta_uqt_slope", -0.022),
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
