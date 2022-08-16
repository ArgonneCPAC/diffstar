"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
import numpy as np
from ..utils import _sigmoid, _inverse_sigmoid

_Q_PARAM_BOUNDS = OrderedDict(
    u_lg_qt=(0.1, 2.0), u_lg_qs=(-3.0, -0.01), u_lg_drop=(-3, 0.0), u_lg_rejuv=(-3, 0.0)
)


def calculate_sigmoid_bounds(param_bounds):
    bounds_out = OrderedDict()

    for key in param_bounds:
        _bounds = (
            float(np.mean(param_bounds[key])),
            abs(float(4.0 / np.diff(param_bounds[key]))),
        )
        bounds_out[key] = _bounds + param_bounds[key]
    return bounds_out


Q_PARAM_BOUNDS = calculate_sigmoid_bounds(_Q_PARAM_BOUNDS)


@jjit
def quenching_function(lgt, u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv):
    """Quenching function halting the star formation of main sequence galaxies.

    After some time, galaxies might experience a rejuvenated star formation.

    Parameters
    ----------
    lgt : ndarray
        Base-10 log cosmic time in Gyr

    u_lg_qt : float
        Unbounded base-10 log of the time at which the quenching event bottoms out,
        i.e., the center of the event, in Gyr units.

    u_lg_qs : float
        Unbounded duration of the quenching event in dex.

    u_lg_drop : float
        Unbounded base-10 log of the lowest drop in SFR.

    u_lg_rejuv : float
        Unbounded base-10 log of the asymptotic SFR value after rejuvenation completes

    Returns
    -------
    History of the multiplicative change of SFR

    """
    lg_qt, lg_qs, lg_drop, lg_rejuv = _get_bounded_q_params(
        u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv
    )
    qs = 10**lg_qs
    _bound_params = (lg_qt, qs, lg_drop, lg_rejuv)
    return 10 ** _quenching_kern(lgt, *_bound_params)


@jjit
def _quenching_kern(lgt, lg_qt, q_dt, q_drop, q_rejuv):
    """Base-10 logarithmic drop and symmetric rise in SFR over a time interval.

    Parameters
    ----------
    lgt : ndarray
        Base-10 log cosmic time in Gyr

    lg_qt : float
        Base-10 log of the time at which the quenching event bottoms out,
        i.e., the center of the event, in Gyr units.

    q_dt : float
        Total duration of the quenching event in dex.
        SFR first begins to drop below zero at lgt = lg_qt - q_dt/2
        SFR first attains its asymptotic final value at lgt = lg_qt + q_dt/2

    q_drop : float
        Base-10 log of the lowest drop in SFR.
        The quenching function reaches this lowest point at t = t_q

    q_rejuv : float
        Base-10 log of the asymptotic SFR value after rejuvenation completes

    Returns
    -------
    History of the base-10 logarithmic change in SFR

    """
    qs = q_dt / 12
    f2 = q_drop - q_rejuv
    return _jax_partial_u_tw_kern(lgt, lg_qt, qs, q_drop, f2)


@jjit
def _jax_tw(y):
    v = -5 * y**7 / 69984 + 7 * y**5 / 2592 - 35 * y**3 / 864 + 35 * y / 96 + 0.5
    res = jnp.where(y < -3, 0, v)
    res = jnp.where(y > 3, 1, res)
    return res


@jjit
def _jax_tw_cuml_kern(x, m, h):
    y = (x - m) / h
    return _jax_tw(y)


@jjit
def _jax_tw_qfunc_kern(lgt, lgqt, tw_h, lgdq):
    tw_m = 3 * tw_h + lgqt
    log_sfr_drop = lgdq * _jax_tw_cuml_kern(lgt, tw_m, tw_h)
    return log_sfr_drop


@jjit
def _jax_partial_u_tw_kern(x, m, h, f1, f2):
    y = (x - m) / h
    z = f1 * _jax_tw(y + 3)
    w = f1 - f2 * _jax_tw(y - 3)
    return jnp.where(y < 0, z, w)


@jjit
def _get_bounded_q_params(u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv):
    lg_qt = _sigmoid(u_lg_qt, *Q_PARAM_BOUNDS["u_lg_qt"])
    qs = _sigmoid(u_lg_qs, *Q_PARAM_BOUNDS["u_lg_qs"])
    lg_drop = _sigmoid(u_lg_drop, *Q_PARAM_BOUNDS["u_lg_drop"])
    lg_rejuv = _get_bounded_lg_rejuv(u_lg_rejuv, lg_drop)
    return lg_qt, qs, lg_drop, lg_rejuv


@jjit
def _get_bounded_lg_drop(u_lg_drop):
    lg_drop = _sigmoid(u_lg_drop, *Q_PARAM_BOUNDS["u_lg_drop"])
    return lg_drop


@jjit
def _get_unbounded_lg_drop(lg_drop):
    u_lg_drop = _inverse_sigmoid(lg_drop, *Q_PARAM_BOUNDS["u_lg_drop"])
    return u_lg_drop


@jjit
def _get_bounded_lg_rejuv(u_lg_rejuv, lg_drop):
    lg_rejuv = _sigmoid(
        u_lg_rejuv,
        *Q_PARAM_BOUNDS["u_lg_rejuv"][:2],
        lg_drop,
        Q_PARAM_BOUNDS["u_lg_rejuv"][3]
    )
    return lg_rejuv


@jjit
def _get_bounded_qt(u_lg_qt):
    lg_qt = _sigmoid(u_lg_qt, *Q_PARAM_BOUNDS["u_lg_qt"])
    return lg_qt


@jjit
def _get_unbounded_q_params(lg_qt, lg_qs, lg_drop, lg_rejuv):
    u_lg_qt = _inverse_sigmoid(lg_qt, *Q_PARAM_BOUNDS["u_lg_qt"])
    u_lg_qs = _inverse_sigmoid(lg_qs, *Q_PARAM_BOUNDS["u_lg_qs"])
    u_lg_drop = _inverse_sigmoid(lg_drop, *Q_PARAM_BOUNDS["u_lg_drop"])
    u_lg_rejuv = _get_unbounded_qrejuv(lg_rejuv, lg_drop)
    return u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv


_get_bounded_q_params_vmap = jjit(vmap(_get_bounded_q_params, (0,) * 4, 0))
_get_unbounded_q_params_vmap = jjit(vmap(_get_unbounded_q_params, (0,) * 4, 0))


@jjit
def _get_unbounded_qrejuv(lg_rejuv, lg_drop):
    u_lg_rejuv = _inverse_sigmoid(
        lg_rejuv,
        *Q_PARAM_BOUNDS["u_lg_rejuv"][:2],
        lg_drop,
        Q_PARAM_BOUNDS["u_lg_rejuv"][3]
    )
    return u_lg_rejuv
