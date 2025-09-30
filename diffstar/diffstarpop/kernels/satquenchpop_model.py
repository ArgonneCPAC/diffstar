"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_Q_SPEED = 5.0
DEFAULT_LGMH_K = 5.0
LGMU_SPEED = 10.0

DEFAULT_SATQUENCH_PDICT = OrderedDict(t_delay=1.0, qprob_hi=0.75)
SatQuenchParams = namedtuple("SatQuenchParams", DEFAULT_SATQUENCH_PDICT.keys())
DEFAULT_SATQUENCH_PARAMS = SatQuenchParams(**DEFAULT_SATQUENCH_PDICT)


DEFAULT_SATQUENCHPOP_PDICT = OrderedDict(
    qp_lgmh_crit=13.0,
    td_lgmhc=13.0,
    td_mlo=2.0,
    td_mhi=-0.5,
    qphi_lgmu_crit=-1.5,
    lgmu_lo_mh_lo=0.1,
    lgmu_hi_mh_lo=0.5,
    lgmu_lo_mh_hi=0.5,
    lgmu_hi_mh_hi=0.9,
)
SatQuenchPopParams = namedtuple("SatQuenchPopParams", DEFAULT_SATQUENCHPOP_PDICT.keys())
DEFAULT_SATQUENCHPOP_PARAMS = SatQuenchPopParams(**DEFAULT_SATQUENCHPOP_PDICT)

LGMH_BOUNDS = (11.0, 15.0)
LGMU_BOUNDS = (-4.0, -0.1)
TDELAY_BOUNDS = (-5.0, 10.0)
QPROB_BOUNDS = (0.0, 1.0)

SATQUENCHPOP_PBOUNDS_PDICT = OrderedDict(
    qp_lgmh_crit=LGMH_BOUNDS,
    td_lgmhc=LGMH_BOUNDS,
    td_mlo=TDELAY_BOUNDS,
    td_mhi=TDELAY_BOUNDS,
    qphi_lgmu_crit=LGMU_BOUNDS,
    lgmu_lo_mh_lo=QPROB_BOUNDS,
    lgmu_hi_mh_lo=QPROB_BOUNDS,
    lgmu_lo_mh_hi=QPROB_BOUNDS,
    lgmu_hi_mh_hi=QPROB_BOUNDS,
)
SATQUENCHPOP_PBOUNDS = SatQuenchPopParams(**SATQUENCHPOP_PBOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in DEFAULT_SATQUENCHPOP_PDICT.keys()]
SatQuenchPopUParams = namedtuple("SatQuenchPopUParams", _UPNAMES)


@jjit
def get_qprob_sat(
    satquenchpop_params, lgmu_infall, logmhost_infall, gyr_since_infall, qprob_cen
):
    frac_delta_qprob = predict_sat_frac_delta_qprob(
        satquenchpop_params, lgmu_infall, logmhost_infall, gyr_since_infall
    )
    delta_qprob = frac_delta_qprob * (1 - qprob_cen)
    qprob_sat = qprob_cen + delta_qprob
    return qprob_sat


@jjit
def predict_sat_frac_delta_qprob(
    satquenchpop_params, lgmu_infall, logmhost_infall, gyr_since_infall
):
    qprob_hi = _qprob_hi_vs_sub_host_mass(
        satquenchpop_params, lgmu_infall, logmhost_infall
    )
    t_delay = _t_delay_vs_logmhost(satquenchpop_params, logmhost_infall)

    sat_quench_params = SatQuenchParams(t_delay, qprob_hi)

    frac_delta_qprob = _satquench_kernel(gyr_since_infall, sat_quench_params)
    return frac_delta_qprob


@jjit
def _qprob_hi_vs_sub_host_mass(satquenchpop_params, lgmu_infall, logmhost_infall):
    qprob_hi_mh_lo = _sigmoid(
        lgmu_infall,
        satquenchpop_params.qphi_lgmu_crit,
        LGMU_SPEED,
        satquenchpop_params.lgmu_lo_mh_lo,
        satquenchpop_params.lgmu_hi_mh_lo,
    )

    qprob_hi_mh_hi = _sigmoid(
        lgmu_infall,
        satquenchpop_params.qphi_lgmu_crit,
        LGMU_SPEED,
        satquenchpop_params.lgmu_lo_mh_hi,
        satquenchpop_params.lgmu_hi_mh_hi,
    )

    qprob_hi = _sigmoid(
        logmhost_infall,
        satquenchpop_params.qp_lgmh_crit,
        DEFAULT_LGMH_K,
        qprob_hi_mh_lo,
        qprob_hi_mh_hi,
    )
    return qprob_hi


@jjit
def _satquench_kernel(gyr_since_infall, sat_quench_params):
    sat_quench_params = SatQuenchParams(*sat_quench_params)
    frac_delta_qprob = _sigmoid(
        gyr_since_infall,
        sat_quench_params.t_delay,
        DEFAULT_Q_SPEED,
        0.0,
        sat_quench_params.qprob_hi,
    )
    return frac_delta_qprob


@jjit
def _t_delay_vs_logmhost(satquenchpop_params, logmhost_infall):
    t_delay_gyr = _sigmoid(
        logmhost_infall,
        satquenchpop_params.td_lgmhc,
        DEFAULT_LGMH_K,
        satquenchpop_params.td_mlo,
        satquenchpop_params.td_mhi,
    )
    return t_delay_gyr


@jjit
def _get_bounded_satq_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_satq_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_satq_params_kern = jjit(vmap(_get_bounded_satq_param, in_axes=_C))
_get_satq_u_params_kern = jjit(vmap(_get_unbounded_satq_param, in_axes=_C))


@jjit
def get_bounded_satquenchpop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _UPNAMES])
    params = _get_satq_params_kern(jnp.array(u_params), jnp.array(SATQUENCHPOP_PBOUNDS))
    return SatQuenchPopParams(*params)


@jjit
def get_unbounded_satquenchpop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_SATQUENCHPOP_PARAMS._fields]
    )
    u_params = _get_satq_u_params_kern(
        jnp.array(params), jnp.array(SATQUENCHPOP_PBOUNDS)
    )
    return SatQuenchPopUParams(*u_params)


DEFAULT_SATQUENCHPOP_U_PARAMS = SatQuenchPopUParams(
    *get_unbounded_satquenchpop_params(DEFAULT_SATQUENCHPOP_PARAMS)
)
