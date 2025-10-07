# sfh_pdf_cen_sat.py
"""Model variant: main-sequence block keyed by (cen, sat) instead of (mseq, qseq)."""

from collections import OrderedDict, namedtuple

from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import (
    _inverse_sigmoid,
    _sigmoid,
    covariance_from_correlation,
    smoothly_clipped_line,
)

# --- constants (unchanged) ---
TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0 = 12.5
LGM_K = 2.0
LGMCRIT_K = 4.0
BOUNDING_K = 0.1

# --- mean parameter dicts ---
# Swap MS block tags: mseq -> cen, qseq (MS part) -> sat
SFH_PDF_CEN_SAT_MU_PDICT = OrderedDict(
    # main-sequence block for centrals
    mean_ulgm_cen_xtp=12.027,
    mean_ulgm_cen_ytp=12.030,
    mean_ulgm_cen_lo=0.901,
    mean_ulgm_cen_hi=0.104,
    mean_ulgy_cen_int=-9.50,
    mean_ulgy_cen_slp=0.43,
    mean_ul_cen_int=-0.75,
    mean_ul_cen_slp=0.80,
    mean_uh_cen_int=-2.04,
    mean_uh_cen_slp=-3.04,
    # main-sequence block for satellites (formerly "qseq" MS block)
    mean_ulgm_sat_xtp=12.246,
    mean_ulgm_sat_ytp=12.200,
    mean_ulgm_sat_lo=0.812,
    mean_ulgm_sat_hi=0.094,
    mean_ulgy_sat_int=-9.50,
    mean_ulgy_sat_slp=0.43,
    mean_ul_sat_int=-0.75,
    mean_ul_sat_slp=0.80,
    mean_uh_sat_int=-2.04,
    mean_uh_sat_slp=-3.04,
    # quenched-only means (unchanged tag names)
    mean_uqt_int=0.96,
    mean_uqt_slp=-0.20,
    mean_uqs_int=-0.16,
    mean_uqs_slp=0.47,
    mean_udrop_int=-2.05,
    mean_udrop_slp=0.18,
    mean_urej_int=-0.97,
    mean_urej_slp=-0.06,
)

SFH_PDF_CEN_SAT_MU_BOUNDS_PDICT = OrderedDict(
    # central MS block bounds
    mean_ulgm_cen_xtp=(11.0, 14.0),
    mean_ulgm_cen_ytp=(11.0, 14.0),
    mean_ulgm_cen_lo=(-1.0, 5.0),
    mean_ulgm_cen_hi=(-5.0, 1.0),
    mean_ulgy_cen_int=(-13.0, -7.0),
    mean_ulgy_cen_slp=(-20.0, 20.0),
    mean_ul_cen_int=(-3.0, 5.0),
    mean_ul_cen_slp=(-20.0, 20.0),
    mean_uh_cen_int=(-5.0, 3.0),
    mean_uh_cen_slp=(-20.0, 20.0),
    # satellite MS block bounds
    mean_ulgm_sat_xtp=(11.0, 14.0),
    mean_ulgm_sat_ytp=(11.0, 14.0),
    mean_ulgm_sat_lo=(-1.0, 5.0),
    mean_ulgm_sat_hi=(-5.0, 1.0),
    mean_ulgy_sat_int=(-13.0, -7.0),
    mean_ulgy_sat_slp=(-20.0, 20.0),
    mean_ul_sat_int=(-3.0, 5.0),
    mean_ul_sat_slp=(-20.0, 20.0),
    mean_uh_sat_int=(-5.0, 3.0),
    mean_uh_sat_slp=(-20.0, 20.0),
    # quenched-only bounds (unchanged)
    mean_uqt_int=(0.0, 2.0),
    mean_uqt_slp=(-20.0, 20.0),
    mean_uqs_int=(-5.0, 2.0),
    mean_uqs_slp=(-20.0, 20.0),
    mean_udrop_int=(-3.0, 2.0),
    mean_udrop_slp=(-20.0, 20.0),
    mean_urej_int=(-10.0, 2.0),
    mean_urej_slp=(-20.0, 20.0),
)

# --- cov parameter dicts for MS block ---
# Swap tags: mseq -> cen, qseq -> sat
SFH_PDF_CEN_SAT_COV_MS_BLOCK_PDICT = OrderedDict(
    # central MS block stds
    std_ulgm_cen_int=0.325,
    std_ulgm_cen_slp=-0.028,
    std_ulgy_cen_int=0.238,
    std_ulgy_cen_slp=-0.004,
    std_ul_cen_int=0.345,
    std_ul_cen_slp=0.008,
    std_uh_cen_int=0.345,
    std_uh_cen_slp=0.008,
    # satellite MS block stds
    std_ulgm_sat_int=0.243,
    std_ulgm_sat_slp=-0.037,
    std_ulgy_sat_int=0.327,
    std_ulgy_sat_slp=-0.082,
    std_ul_sat_int=0.210,
    std_ul_sat_slp=0.271,
    std_uh_sat_int=0.210,
    std_uh_sat_slp=0.271,
)

SFH_PDF_CEN_SAT_COV_MS_BLOCK_BOUNDS_PDICT = OrderedDict(
    # central
    std_ulgm_cen_int=(0.01, 1.0),
    std_ulgm_cen_slp=(-1.00, 1.0),
    std_ulgy_cen_int=(0.01, 1.0),
    std_ulgy_cen_slp=(-1.00, 1.0),
    std_ul_cen_int=(0.01, 3.0),
    std_ul_cen_slp=(-1.00, 1.0),
    std_uh_cen_int=(0.01, 3.0),
    std_uh_cen_slp=(-1.00, 1.0),
    # satellite
    std_ulgm_sat_int=(0.01, 1.0),
    std_ulgm_sat_slp=(-1.00, 1.0),
    std_ulgy_sat_int=(0.01, 1.0),
    std_ulgy_sat_slp=(-1.00, 1.0),
    std_ul_sat_int=(0.01, 3.0),
    std_ul_sat_slp=(-1.00, 1.0),
    std_uh_sat_int=(0.01, 3.0),
    std_uh_sat_slp=(-1.00, 1.0),
)

# --- quenched-block cov dicts (unchanged) ---
SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    std_uqt_int=0.070,
    std_uqt_slp=-0.045,
    std_uqs_int=0.444,
    std_uqs_slp=-0.300,
    std_udrop_int=0.779,
    std_udrop_slp=-0.166,
    std_urej_int=1.538,
    std_urej_slp=-0.018,
)
SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_uqt_int=(0.01, 0.5),
    std_uqt_slp=(-1.00, 1.0),
    std_uqs_int=(0.01, 1.0),
    std_uqs_slp=(-1.00, 1.0),
    std_udrop_int=(0.01, 2.0),
    std_udrop_slp=(-1.00, 1.0),
    std_urej_int=(0.01, 2.0),
    std_urej_slp=(-1.00, 1.0),
)

# --- fraction of quenched (unchanged) ---
SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    frac_quench_cen_x0_tpeak=7.0,
    frac_quench_cen_k_tpeak=2.0,
    frac_quench_cen_x0_ylotpeak=13.0,
    frac_quench_cen_x0_yhitpeak=12.0,
    frac_quench_cen_ylo_ylotpeak=0.65,
    frac_quench_cen_ylo_yhitpeak=0.05,
    frac_quench_cen_k=3.848,
    frac_quench_cen_yhi=0.971,
    frac_quench_sat_x0_tpeak=7.0,
    frac_quench_sat_k_tpeak=2.0,
    frac_quench_sat_x0_ylotpeak=13.0,
    frac_quench_sat_x0_yhitpeak=12.0,
    frac_quench_sat_ylo_ylotpeak=0.65,
    frac_quench_sat_ylo_yhitpeak=0.05,
    frac_quench_sat_k=3.848,
    frac_quench_sat_yhi=0.971,
)

SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_cen_x0_tpeak=(1.0, 14.0),
    frac_quench_cen_k_tpeak=(0.01, 10.0),
    frac_quench_cen_x0_ylotpeak=(11.0, 14.0),
    frac_quench_cen_x0_yhitpeak=(11.0, 14.0),
    frac_quench_cen_ylo_ylotpeak=(0.0, 1.0),
    frac_quench_cen_ylo_yhitpeak=(0.0, 1.0),
    frac_quench_cen_k=(0.01, 5.0),
    frac_quench_cen_yhi=(0.0, 1.0),
    frac_quench_sat_x0_tpeak=(1.0, 14.0),
    frac_quench_sat_k_tpeak=(0.01, 10.0),
    frac_quench_sat_x0_ylotpeak=(11.0, 14.0),
    frac_quench_sat_x0_yhitpeak=(11.0, 14.0),
    frac_quench_sat_ylo_ylotpeak=(0.0, 1.0),
    frac_quench_sat_ylo_yhitpeak=(0.0, 1.0),
    frac_quench_sat_k=(0.01, 5.0),
    frac_quench_sat_yhi=(0.0, 1.0),
)

# --- shared bounding dicts (unchanged) ---
BOUNDING_MEAN_VALS_PDICT = OrderedDict(
    mean_ulgm=(11.0, 13.0),
    mean_ulgy=(-13.0, -7.0),
    mean_ul=(-3.0, 5.0),
    mean_uh=(-5.0, 3.0),
    mean_uqt=(0.0, 2.0),
    mean_uqs=(-5.0, 2.0),
    mean_udrop=(-3.0, 2.0),
    mean_urej=(-10.0, 2.0),
)
BOUNDING_STD_VALS_PDICT = OrderedDict(
    std_ulgm=(0.01, 1.0),
    std_ulgy=(0.01, 1.0),
    std_ul=(0.01, 3.0),
    std_uh=(0.01, 3.0),
    std_uqt=(0.01, 0.5),
    std_uqs=(0.01, 1.0),
    std_udrop=(0.01, 2.0),
    std_urej=(0.01, 2.0),
)

DELTA_UQT_PDICT = OrderedDict(
    delta_uqt_x0=5.0,
    delta_uqt_k=1.0,
    delta_uqt_ylo=-0.4,
    delta_uqt_yhi=0.1,
    delta_uqt_slope=0.01,
)
DELTA_UQT_BOUNDS_PDICT = OrderedDict(
    delta_uqt_x0=(1.0, 14.0),
    delta_uqt_k=(0.01, 5.0),
    delta_uqt_ylo=(-1.0, 1.0),
    delta_uqt_yhi=(-1.0, 1.0),
    delta_uqt_slope=(-0.1, 0.1),
)

# --- assemble parameter/bounds namedtuples ---
SFH_PDF_CEN_SAT_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_CEN_SAT_PDICT.update(SFH_PDF_CEN_SAT_MU_PDICT)
SFH_PDF_CEN_SAT_PDICT.update(SFH_PDF_CEN_SAT_COV_MS_BLOCK_PDICT)
SFH_PDF_CEN_SAT_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)
SFH_PDF_CEN_SAT_PDICT.update(DELTA_UQT_PDICT)

SFH_PDF_CEN_SAT_BOUNDS_PDICT = SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT.copy()
SFH_PDF_CEN_SAT_BOUNDS_PDICT.update(SFH_PDF_CEN_SAT_MU_BOUNDS_PDICT)
SFH_PDF_CEN_SAT_BOUNDS_PDICT.update(SFH_PDF_CEN_SAT_COV_MS_BLOCK_BOUNDS_PDICT)
SFH_PDF_CEN_SAT_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT)
SFH_PDF_CEN_SAT_BOUNDS_PDICT.update(DELTA_UQT_BOUNDS_PDICT)

BOUNDING_VALS_PDICT = BOUNDING_MEAN_VALS_PDICT.copy()
BOUNDING_VALS_PDICT.update(BOUNDING_STD_VALS_PDICT.copy())

QParams = namedtuple("QParams", list(SFH_PDF_CEN_SAT_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QParams(**SFH_PDF_CEN_SAT_PDICT)
SFH_PDF_CEN_SAT_PBOUNDS = QParams(**SFH_PDF_CEN_SAT_BOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in QParams._fields]
QUParams = namedtuple("QUParams", _UPNAMES)

BoundingParams = namedtuple("BoundingParams", list(BOUNDING_VALS_PDICT.keys()))
BOUNDING_VALS = BoundingParams(**BOUNDING_VALS_PDICT)


# --- helpers (unchanged) ---
@jjit
def line_model(x, y0, m, y_lo, y_hi):
    return smoothly_clipped_line(x, LGM_X0, y0, m, y_lo, y_hi)


@jjit
def _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - xtp)


def Mcrit_model(x, xtp, ytp, lo, hi):
    x0 = xtp
    slope_k = 3.0
    return _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi)


# --- kernel (updated MS block targets) ---
@jjit
def _sfh_pdf_scalar_kernel(params, logmp0, tpeak):
    frac_quench_cen = _frac_quench_cen(params, logmp0, tpeak)
    frac_quench_sat = _frac_quench_sat(params, logmp0, tpeak)

    # main-sequence block means for centrals & satellites
    mu_cen_ms_block = _get_mean_u_params_cen(params, logmp0)
    mu_sat_ms_block = _get_mean_u_params_sat(params, logmp0)
    # quenched means (unchanged tag names/structure)
    mu_qseq = _get_mean_u_params_qseq(params, logmp0, tpeak)

    cov_cen_ms_block = _get_covariance_cen_ms_block(params, logmp0)
    cov_sat_ms_block = _get_covariance_sat_ms_block(params, logmp0)
    cov_qseq_q_block = _get_covariance_qseq_q_block(params, logmp0)

    return (
        frac_quench_cen,
        frac_quench_sat,
        mu_cen_ms_block,
        mu_sat_ms_block,
        mu_qseq,
        cov_cen_ms_block,
        cov_sat_ms_block,
        cov_qseq_q_block,
    )


# --- delta_uqt and means ---
@jjit
def _delta_uqt(params, logmp0, tpeak):
    delta_uqt_vs_tpeak = _sigmoid(
        tpeak,
        params.delta_uqt_x0,
        params.delta_uqt_k,
        params.delta_uqt_ylo,
        params.delta_uqt_yhi,
    )
    delta_uqt_vs_logmp0 = (logmp0 - 12.5) * params.delta_uqt_slope
    return delta_uqt_vs_tpeak + delta_uqt_vs_logmp0


@jjit
def _get_mean_u_params_cen(params, logmp0):
    ulgm = Mcrit_model(
        logmp0,
        params.mean_ulgm_cen_xtp,
        params.mean_ulgm_cen_ytp,
        params.mean_ulgm_cen_lo,
        params.mean_ulgm_cen_hi,
    )
    ulgy = line_model(
        logmp0,
        params.mean_ulgy_cen_int,
        params.mean_ulgy_cen_slp,
        *BOUNDING_VALS.mean_ulgy,
    )
    ul = line_model(
        logmp0, params.mean_ul_cen_int, params.mean_ul_cen_slp, *BOUNDING_VALS.mean_ul
    )
    uh = line_model(
        logmp0, params.mean_uh_cen_int, params.mean_uh_cen_slp, *BOUNDING_VALS.mean_uh
    )
    return (ulgm, ulgy, ul, uh)


@jjit
def _get_mean_u_params_sat(params, logmp0):
    ulgm = Mcrit_model(
        logmp0,
        params.mean_ulgm_sat_xtp,
        params.mean_ulgm_sat_ytp,
        params.mean_ulgm_sat_lo,
        params.mean_ulgm_sat_hi,
    )
    ulgy = line_model(
        logmp0,
        params.mean_ulgy_sat_int,
        params.mean_ulgy_sat_slp,
        *BOUNDING_VALS.mean_ulgy,
    )
    ul = line_model(
        logmp0, params.mean_ul_sat_int, params.mean_ul_sat_slp, *BOUNDING_VALS.mean_ul
    )
    uh = line_model(
        logmp0, params.mean_uh_sat_int, params.mean_uh_sat_slp, *BOUNDING_VALS.mean_uh
    )
    return (ulgm, ulgy, ul, uh)


@jjit
def _get_mean_u_params_qseq(params, logmp0, tpeak):
    _uqt = line_model(
        logmp0, params.mean_uqt_int, params.mean_uqt_slp, *BOUNDING_VALS.mean_uqt
    )
    uqt = _uqt + _delta_uqt(params, logmp0, tpeak)
    uqs = line_model(
        logmp0, params.mean_uqs_int, params.mean_uqs_slp, *BOUNDING_VALS.mean_uqs
    )
    udrop = line_model(
        logmp0, params.mean_udrop_int, params.mean_udrop_slp, *BOUNDING_VALS.mean_udrop
    )
    urej = line_model(
        logmp0, params.mean_urej_int, params.mean_urej_slp, *BOUNDING_VALS.mean_urej
    )
    return (uqt, uqs, udrop, urej)


# --- cov helpers for MS block (cen/sat) ---
@jjit
def _get_cov_params_cen_ms_block(params, logmp0):
    std_ulgm = line_model(
        logmp0,
        params.std_ulgm_cen_int,
        params.std_ulgm_cen_slp,
        *BOUNDING_VALS.std_ulgm,
    )
    std_ulgy = line_model(
        logmp0,
        params.std_ulgy_cen_int,
        params.std_ulgy_cen_slp,
        *BOUNDING_VALS.std_ulgy,
    )
    std_ul = line_model(
        logmp0, params.std_ul_cen_int, params.std_ul_cen_slp, *BOUNDING_VALS.std_ul
    )
    std_uh = line_model(
        logmp0, params.std_uh_cen_int, params.std_uh_cen_slp, *BOUNDING_VALS.std_uh
    )
    diags = std_ulgm, std_ulgy, std_ul, std_uh
    off_diags = jnp.zeros(6).astype(float)
    return diags, off_diags


@jjit
def _get_cov_params_sat_ms_block(params, logmp0):
    std_ulgm = line_model(
        logmp0,
        params.std_ulgm_sat_int,
        params.std_ulgm_sat_slp,
        *BOUNDING_VALS.std_ulgm,
    )
    std_ulgy = line_model(
        logmp0,
        params.std_ulgy_sat_int,
        params.std_ulgy_sat_slp,
        *BOUNDING_VALS.std_ulgy,
    )
    std_ul = line_model(
        logmp0, params.std_ul_sat_int, params.std_ul_sat_slp, *BOUNDING_VALS.std_ul
    )
    std_uh = line_model(
        logmp0, params.std_uh_sat_int, params.std_uh_sat_slp, *BOUNDING_VALS.std_uh
    )
    diags = std_ulgm, std_ulgy, std_ul, std_uh
    off_diags = jnp.zeros(6).astype(float)
    return diags, off_diags


@jjit
def _get_cov_params_qseq_q_block(params, logmp0):
    std_uqt = line_model(
        logmp0, params.std_uqt_int, params.std_uqt_slp, *BOUNDING_VALS.std_uqt
    )
    std_uqs = line_model(
        logmp0, params.std_uqs_int, params.std_uqs_slp, *BOUNDING_VALS.std_uqs
    )
    std_udrop = line_model(
        logmp0, params.std_udrop_int, params.std_udrop_slp, *BOUNDING_VALS.std_udrop
    )
    std_urej = line_model(
        logmp0, params.std_urej_int, params.std_urej_slp, *BOUNDING_VALS.std_urej
    )
    diags = std_uqt, std_uqs, std_udrop, std_urej
    off_diags = jnp.zeros(6).astype(float)
    return diags, off_diags


# --- covariance builders ---
@jjit
def _cov_from_diags_off(off_diags, diags):
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    return covariance_from_correlation(corr_matrix, jnp.array(diags))


@jjit
def _get_covariance_cen_ms_block(params, logmp0):
    diags, off_diags = _get_cov_params_cen_ms_block(params, logmp0)
    return _cov_from_diags_off(off_diags, diags)


@jjit
def _get_covariance_sat_ms_block(params, logmp0):
    diags, off_diags = _get_cov_params_sat_ms_block(params, logmp0)
    return _cov_from_diags_off(off_diags, diags)


@jjit
def _get_covariance_qseq_q_block(params, logmp0):
    diags, off_diags = _get_cov_params_qseq_q_block(params, logmp0)
    return _cov_from_diags_off(off_diags, diags)


# --- quenched fraction pieces (unchanged) ---
@jjit
def _frac_quench_cen(params, logmp0, tpeak):
    fquench_x0 = _sigmoid(
        tpeak,
        params.frac_quench_cen_x0_tpeak,
        params.frac_quench_cen_k_tpeak,
        params.frac_quench_cen_x0_ylotpeak,
        params.frac_quench_cen_x0_yhitpeak,
    )
    frac_quench_cen_ylo = _sigmoid(
        tpeak,
        params.frac_quench_cen_x0_tpeak,
        params.frac_quench_cen_k_tpeak,
        params.frac_quench_cen_ylo_ylotpeak,
        params.frac_quench_cen_ylo_yhitpeak,
    )
    return _sigmoid(
        logmp0,
        fquench_x0,
        params.frac_quench_cen_k,
        frac_quench_cen_ylo,
        params.frac_quench_cen_yhi,
    )


@jjit
def _frac_quench_sat(params, logmp0, tpeak):
    fquench_x0 = _sigmoid(
        tpeak,
        params.frac_quench_sat_x0_tpeak,
        params.frac_quench_sat_k_tpeak,
        params.frac_quench_sat_x0_ylotpeak,
        params.frac_quench_sat_x0_yhitpeak,
    )
    frac_quench_sat_ylo = _sigmoid(
        tpeak,
        params.frac_quench_sat_x0_tpeak,
        params.frac_quench_sat_k_tpeak,
        params.frac_quench_sat_ylo_ylotpeak,
        params.frac_quench_sat_ylo_yhitpeak,
    )
    return _sigmoid(
        logmp0,
        fquench_x0,
        params.frac_quench_sat_k,
        frac_quench_sat_ylo,
        params.frac_quench_sat_yhi,
    )


# --- bounded/unbounded transforms (unchanged mechanics) ---
@jjit
def _get_p_from_u_p_scalar(u_p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    return _sigmoid(u_p, p0, BOUNDING_K, lo, hi)


@jjit
def _get_u_p_from_p_scalar(p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    return _inverse_sigmoid(p, p0, BOUNDING_K, lo, hi)


_get_p_from_u_p_vmap = jjit(vmap(_get_p_from_u_p_scalar, in_axes=(0, 0)))
_get_u_p_from_p_vmap = jjit(vmap(_get_u_p_from_p_scalar, in_axes=(0, 0)))


@jjit
def get_bounded_sfh_pdf_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in QUParams._fields])
    params = _get_p_from_u_p_vmap(
        jnp.array(u_params), jnp.array(SFH_PDF_CEN_SAT_PBOUNDS)
    )
    return QParams(*params)


def get_unbounded_sfh_pdf_params(params):
    params = jnp.array([getattr(params, pname) for pname in QParams._fields])
    u_params = _get_u_p_from_p_vmap(
        jnp.array(params), jnp.array(SFH_PDF_CEN_SAT_PBOUNDS)
    )
    return QUParams(*u_params)


SFH_PDF_QUENCH_U_PARAMS = QUParams(*get_unbounded_sfh_pdf_params(SFH_PDF_QUENCH_PARAMS))
