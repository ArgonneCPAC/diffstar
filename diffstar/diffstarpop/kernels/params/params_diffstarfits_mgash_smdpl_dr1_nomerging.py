from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.113),
        ("mean_ulgm_mseq_ytp", 12.089),
        ("mean_ulgm_mseq_lo", 0.821),
        ("mean_ulgm_mseq_hi", 0.108),
        ("mean_ulgy_mseq_int", -9.899),
        ("mean_ulgy_mseq_slp", 0.389),
        ("mean_ul_mseq_int", -0.804),
        ("mean_ul_mseq_slp", 0.743),
        ("mean_uh_mseq_int", -1.615),
        ("mean_uh_mseq_slp", -2.708),
        ("mean_ulgm_qseq_xtp", 12.282),
        ("mean_ulgm_qseq_ytp", 12.238),
        ("mean_ulgm_qseq_lo", 0.781),
        ("mean_ulgm_qseq_hi", 0.072),
        ("mean_ulgy_qseq_int", -9.977),
        ("mean_ulgy_qseq_slp", 0.734),
        ("mean_ul_qseq_int", -0.705),
        ("mean_ul_qseq_slp", 0.850),
        ("mean_uh_qseq_int", -0.350),
        ("mean_uh_qseq_slp", -1.369),
        ("mean_uqt_int", 0.950),
        ("mean_uqt_slp", -0.203),
        ("mean_uqs_int", -0.112),
        ("mean_uqs_slp", 0.395),
        ("mean_udrop_int", -2.143),
        ("mean_udrop_slp", 0.232),
        ("mean_urej_int", -1.227),
        ("mean_urej_slp", -0.010),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.287),
        ("std_ulgm_mseq_slp", 0.146),
        ("std_ulgy_mseq_int", 0.285),
        ("std_ulgy_mseq_slp", 0.039),
        ("std_ul_mseq_int", 1.263),
        ("std_ul_mseq_slp", 0.445),
        ("std_uh_mseq_int", 2.516),
        ("std_uh_mseq_slp", -0.900),
        ("std_ulgm_qseq_int", 0.250),
        ("std_ulgm_qseq_slp", 0.190),
        ("std_ulgy_qseq_int", 0.307),
        ("std_ulgy_qseq_slp", 0.022),
        ("std_ul_qseq_int", 1.687),
        ("std_ul_qseq_slp", 0.269),
        ("std_uh_qseq_int", 1.499),
        ("std_uh_qseq_slp", -0.246),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.121),
        ("std_uqt_slp", 0.052),
        ("std_uqs_int", 0.567),
        ("std_uqs_slp", -0.029),
        ("std_udrop_int", 0.749),
        ("std_udrop_slp", 0.162),
        ("std_urej_int", 1.357),
        ("std_urej_slp", 0.069),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.652),
        ("frac_quench_cen_x0_yhitpeak", 11.882),
        ("frac_quench_cen_ylo_ylotpeak", 0.555),
        ("frac_quench_cen_ylo_yhitpeak", 0.010),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.652),
        ("frac_quench_sat_x0_yhitpeak", 11.882),
        ("frac_quench_sat_ylo_ylotpeak", 0.555),
        ("frac_quench_sat_ylo_yhitpeak", 0.010),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.554),
        ("delta_uqt_k", 0.501),
        ("delta_uqt_ylo", -0.578),
        ("delta_uqt_yhi", 0.037),
        ("delta_uqt_slope", -0.050),
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


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array


DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS
)
