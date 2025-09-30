from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_mgash import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.315),
        ("mean_ulgm_mseq_ytp", 11.579),
        ("mean_ulgm_mseq_lo", 0.122),
        ("mean_ulgm_mseq_hi", 0.372),
        ("mean_ulgy_mseq_int", -9.593),
        ("mean_ulgy_mseq_slp", 0.403),
        ("mean_ul_mseq_int", -0.358),
        ("mean_ul_mseq_slp", 1.187),
        ("mean_uh_mseq_int", -0.473),
        ("mean_uh_mseq_slp", 1.422),
        ("mean_ulgm_qseq_xtp", 11.803),
        ("mean_ulgm_qseq_ytp", 11.551),
        ("mean_ulgm_qseq_lo", -0.008),
        ("mean_ulgm_qseq_hi", 0.282),
        ("mean_ulgy_qseq_int", -9.828),
        ("mean_ulgy_qseq_slp", 0.562),
        ("mean_ul_qseq_int", -0.328),
        ("mean_ul_qseq_slp", 1.712),
        ("mean_uh_qseq_int", -0.755),
        ("mean_uh_qseq_slp", -0.501),
        ("mean_uqt_int", 0.990),
        ("mean_uqt_slp", -0.071),
        ("mean_uqs_int", 0.172),
        ("mean_uqs_slp", -0.124),
        ("mean_udrop_int", -2.095),
        ("mean_udrop_slp", 0.646),
        ("mean_urej_int", -0.979),
        ("mean_urej_slp", 0.478),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.404),
        ("std_ulgm_mseq_slp", -0.013),
        ("std_ulgy_mseq_int", 0.281),
        ("std_ulgy_mseq_slp", 0.009),
        ("std_ul_mseq_int", 1.583),
        ("std_ul_mseq_slp", 0.462),
        ("std_uh_mseq_int", 1.616),
        ("std_uh_mseq_slp", -0.297),
        ("std_ulgm_qseq_int", 0.361),
        ("std_ulgm_qseq_slp", 0.056),
        ("std_ulgy_qseq_int", 0.275),
        ("std_ulgy_qseq_slp", -0.011),
        ("std_ul_qseq_int", 1.740),
        ("std_ul_qseq_slp", -0.135),
        ("std_uh_qseq_int", 1.327),
        ("std_uh_qseq_slp", -0.477),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.124),
        ("std_uqt_slp", 0.082),
        ("std_uqs_int", 0.577),
        ("std_uqs_slp", 0.204),
        ("std_udrop_int", 0.806),
        ("std_udrop_slp", -0.100),
        ("std_urej_int", 1.197),
        ("std_urej_slp", -0.076),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 11.610),
        ("frac_quench_cen_x0_yhitpeak", 12.066),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.099),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 11.610),
        ("frac_quench_sat_x0_yhitpeak", 12.066),
        ("frac_quench_sat_ylo_ylotpeak", 0.990),
        ("frac_quench_sat_ylo_yhitpeak", 0.099),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 3.294),
        ("delta_uqt_k", 4.727),
        ("delta_uqt_ylo", -0.326),
        ("delta_uqt_yhi", 0.025),
        ("delta_uqt_slope", 0.021),
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


DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS
)
