from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash_ecrit import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.097),
        ("mean_ulgm_mseq_ytp", 12.082),
        ("mean_ulgm_mseq_lo", 0.812),
        ("mean_ulgm_mseq_hi", 0.185),
        ("mean_ulgy_mseq_xtp", 12.131),
        ("mean_ulgy_mseq_ytp", -10.094),
        ("mean_ulgy_mseq_lo", 0.929),
        ("mean_ulgy_mseq_hi", 0.400),
        ("mean_ul_mseq_int", -1.815),
        ("mean_ul_mseq_slp", 1.089),
        ("mean_uh_mseq_int", 0.820),
        ("mean_uh_mseq_slp", -0.458),
        ("mean_ulgm_qseq_xtp", 12.168),
        ("mean_ulgm_qseq_ytp", 12.153),
        ("mean_ulgm_qseq_lo", 0.882),
        ("mean_ulgm_qseq_hi", 0.040),
        ("mean_ulgy_qseq_xtp", 12.803),
        ("mean_ulgy_qseq_ytp", -9.594),
        ("mean_ulgy_qseq_lo", 0.714),
        ("mean_ulgy_qseq_hi", 0.400),
        ("mean_ul_qseq_int", -2.700),
        ("mean_ul_qseq_slp", -0.220),
        ("mean_uh_qseq_int", 1.910),
        ("mean_uh_qseq_slp", 1.035),
        ("mean_uqt_int", 1.026),
        ("mean_uqt_slp", 0.011),
        ("mean_uqs_int", -0.230),
        ("mean_uqs_slp", 0.104),
        ("mean_udrop_int", -1.814),
        ("mean_udrop_slp", 0.054),
        ("mean_urej_int", -0.739),
        ("mean_urej_slp", -0.261),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.254),
        ("std_ulgm_mseq_slp", 0.153),
        ("std_ulgy_mseq_int", 0.273),
        ("std_ulgy_mseq_slp", 0.027),
        ("std_ul_mseq_int", 1.941),
        ("std_ul_mseq_slp", 0.516),
        ("std_uh_mseq_int", 1.458),
        ("std_uh_mseq_slp", 0.020),
        ("std_ulgm_qseq_int", 0.194),
        ("std_ulgm_qseq_slp", 0.096),
        ("std_ulgy_qseq_int", 0.291),
        ("std_ulgy_qseq_slp", -0.223),
        ("std_ul_qseq_int", 2.091),
        ("std_ul_qseq_slp", -0.183),
        ("std_uh_qseq_int", 1.017),
        ("std_uh_qseq_slp", 0.066),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.079),
        ("std_uqt_slp", 0.001),
        ("std_uqs_int", 0.615),
        ("std_uqs_slp", 0.217),
        ("std_udrop_int", 0.798),
        ("std_udrop_slp", -0.124),
        ("std_urej_int", 1.168),
        ("std_urej_slp", -0.025),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.280),
        ("frac_quench_cen_x0_yhitpeak", 11.658),
        ("frac_quench_cen_ylo_ylotpeak", 0.010),
        ("frac_quench_cen_ylo_yhitpeak", 0.040),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.280),
        ("frac_quench_sat_x0_yhitpeak", 11.658),
        ("frac_quench_sat_ylo_ylotpeak", 0.010),
        ("frac_quench_sat_ylo_yhitpeak", 0.040),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 3.260),
        ("delta_uqt_k", 0.536),
        ("delta_uqt_ylo", -0.372),
        ("delta_uqt_yhi", 0.015),
        ("delta_uqt_slope", -0.065),
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


DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
