"""
"""
from collections import OrderedDict, namedtuple

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_Q_PDICT = OrderedDict(
    lg_qt=1.0, qlglgdt=-0.50725, lg_drop=-1.01773, lg_rejuv=-0.212307
)
QParams = namedtuple("QParams", list(DEFAULT_Q_PDICT.keys()))

DEFAULT_Q_PARAMS = QParams(*list(DEFAULT_Q_PDICT.values()))

Q_PARAM_BOUNDS_PDICT = OrderedDict(
    lg_qt=(0.1, 2.0), qlglgdt=(-3.0, -0.01), lg_drop=(-3, 0.0), lg_rejuv=(-3, 0.0)
)


def calculate_sigmoid_bounds(param_bounds):
    bounds_out = OrderedDict()

    for key in param_bounds:
        _bounds = (
            float(np.mean(param_bounds[key])),
            abs(float(4.0 / np.diff(param_bounds[key])[0])),
        )
        bounds_out[key] = _bounds + param_bounds[key]
    return bounds_out


Q_BOUNDING_SIGMOID_PDICT = calculate_sigmoid_bounds(Q_PARAM_BOUNDS_PDICT)


@jjit
def _quenching_kern_u_params(lgt, u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv):
    """Quenching function halting the star formation of main sequence galaxies.

    After some time, galaxies might experience a rejuvenated star formation.

    Parameters
    ----------
    lgt : ndarray
        Base-10 log cosmic time in Gyr

    u_lg_qt : float
        Unbounded base-10 log of the time at which the quenching event bottoms out,
        i.e., the center of the event, in Gyr units.

    u_qlglgdt : float
        Unbounded value of log10(log10(q_dt))
        Controls duration of quenching event

    u_lg_drop : float
        Unbounded base-10 log of the lowest drop in SFR.

    u_lg_rejuv : float
        Unbounded base-10 log of the asymptotic SFR value after rejuvenation completes

    Returns
    -------
    History of the multiplicative change of SFR

    """
    lg_qt, qlglgdt, lg_drop, lg_rejuv = _get_bounded_q_params(
        u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv
    )
    lg_q_dt = 10**qlglgdt
    _bound_params = (lg_qt, lg_q_dt, lg_drop, lg_rejuv)
    return _quenching_kern(lgt, *_bound_params)


@jjit
def _quenching_kern(lgt, lg_qt, lg_q_dt, q_drop, q_rejuv):
    """Base-10 logarithmic drop and symmetric rise in SFR over a time interval.

    Parameters
    ----------
    lgt : ndarray
        Base-10 log cosmic time in Gyr

    lg_qt : float
        Base-10 log of the time at which the quenching event bottoms out,
        i.e., the center of the event, in Gyr units.

    lg_q_dt : float
        Total duration of the quenching event in dex.
        SFR first begins to drop below zero at lgt = lg_qt - lg_q_dt/2
        SFR first attains its asymptotic final value at lgt = lg_qt + lg_q_dt/2

    q_drop : float
        Base-10 log of the lowest drop in SFR.
        The quenching function reaches this lowest point at t = t_q

    q_rejuv : float
        Base-10 log of the asymptotic SFR value after rejuvenation completes

    Returns
    -------
    History of the multiplicative change of SFR

    """
    lg_q_dt_by_12 = lg_q_dt / 12  # account for 6σ width of two successive triweights
    f2 = q_drop - q_rejuv
    return 10 ** _jax_partial_u_tw_kern(lgt, lg_qt, lg_q_dt_by_12, q_drop, f2)


@jjit
def _jax_tw(y):
    v = -5 * y**7 / 69984 + 7 * y**5 / 2592 - 35 * y**3 / 864 + 35 * y / 96 + 0.5
    res = jnp.where(y < -3, 0, v)
    res = jnp.where(y > 3, 1, res)
    return res


@jjit
def _jax_partial_u_tw_kern(x, m, h, f1, f2):
    y = (x - m) / h
    z = f1 * _jax_tw(y + 3)
    w = f1 - f2 * _jax_tw(y - 3)
    return jnp.where(y < 0, z, w)


@jjit
def _get_bounded_q_params(u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv):
    lg_qt = _sigmoid(u_lg_qt, *Q_BOUNDING_SIGMOID_PDICT["lg_qt"])
    qlglgdt = _sigmoid(u_qlglgdt, *Q_BOUNDING_SIGMOID_PDICT["qlglgdt"])
    lg_drop = _sigmoid(u_lg_drop, *Q_BOUNDING_SIGMOID_PDICT["lg_drop"])
    lg_rejuv = _get_bounded_lg_rejuv(u_lg_rejuv, lg_drop)
    return lg_qt, qlglgdt, lg_drop, lg_rejuv


@jjit
def _get_bounded_lg_drop(u_lg_drop):
    lg_drop = _sigmoid(u_lg_drop, *Q_BOUNDING_SIGMOID_PDICT["lg_drop"])
    return lg_drop


@jjit
def _get_unbounded_lg_drop(lg_drop):
    u_lg_drop = _inverse_sigmoid(lg_drop, *Q_BOUNDING_SIGMOID_PDICT["lg_drop"])
    return u_lg_drop


@jjit
def _get_bounded_lg_rejuv(u_lg_rejuv, lg_drop):
    lg_rejuv = _sigmoid(
        u_lg_rejuv,
        *Q_BOUNDING_SIGMOID_PDICT["lg_rejuv"][:2],
        lg_drop,
        Q_BOUNDING_SIGMOID_PDICT["lg_rejuv"][3],
    )
    return lg_rejuv


@jjit
def _get_bounded_qt(u_lg_qt):
    lg_qt = _sigmoid(u_lg_qt, *Q_BOUNDING_SIGMOID_PDICT["lg_qt"])
    return lg_qt


@jjit
def _get_unbounded_q_params(lg_qt, qlglgdt, lg_drop, lg_rejuv):
    u_lg_qt = _inverse_sigmoid(lg_qt, *Q_BOUNDING_SIGMOID_PDICT["lg_qt"])
    u_qlglgdt = _inverse_sigmoid(qlglgdt, *Q_BOUNDING_SIGMOID_PDICT["qlglgdt"])
    u_lg_drop = _inverse_sigmoid(lg_drop, *Q_BOUNDING_SIGMOID_PDICT["lg_drop"])
    u_lg_rejuv = _get_unbounded_qrejuv(lg_rejuv, lg_drop)
    return u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv


@jjit
def _get_unbounded_qrejuv(lg_rejuv, lg_drop):
    u_lg_rejuv = _inverse_sigmoid(
        lg_rejuv,
        *Q_BOUNDING_SIGMOID_PDICT["lg_rejuv"][:2],
        lg_drop,
        Q_BOUNDING_SIGMOID_PDICT["lg_rejuv"][3],
    )
    return u_lg_rejuv


@jjit
def _get_bounded_q_params_galpop_kern(q_params):
    return jnp.array(_get_bounded_q_params(*q_params))


@jjit
def _get_unbounded_q_params_galpop_kern(u_q_params):
    return jnp.array(_get_unbounded_q_params(*u_q_params))


_get_bounded_q_params_vmap = jjit(vmap(_get_bounded_q_params_galpop_kern, in_axes=(0,)))
_get_unbounded_q_params_vmap = jjit(
    vmap(_get_unbounded_q_params_galpop_kern, in_axes=(0,))
)


QUParams = namedtuple("QUParams", ["u_" + key for key in DEFAULT_Q_PDICT.keys()])
DEFAULT_U_Q_PARAMS = QUParams(*_get_unbounded_q_params(*DEFAULT_Q_PARAMS))
DEFAULT_U_Q_PDICT = OrderedDict(
    [(key, val) for key, val in zip(DEFAULT_U_Q_PARAMS._fields, DEFAULT_U_Q_PARAMS)]
)

DEFAULT_Q_U_PARAMS_UNQUENCHED = QUParams(*[5] * 4)
DEFAULT_Q_PARAMS_UNQUENCHED = QParams(
    *_get_bounded_q_params(*DEFAULT_Q_U_PARAMS_UNQUENCHED)
)
