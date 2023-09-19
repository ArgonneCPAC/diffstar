"""Model for the quenching of individual galaxies."""

from jax import jit as jjit

from .kernels import quenching_kernels as qk


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
    return qk._quenching_kern_u_params(lgt, u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv)


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
    return qk._quenching_kern(lgt, lg_qt, q_dt, q_drop, q_rejuv)


@jjit
def _jax_tw(y):
    return qk._jax_tw(y)


@jjit
def _jax_tw_cuml_kern(x, m, h):
    return qk._jax_tw_cuml_kern(x, m, h)


@jjit
def _jax_tw_qfunc_kern(lgt, lgqt, tw_h, lgdq):
    return qk._jax_tw_qfunc_kern(lgt, lgqt, tw_h, lgdq)


@jjit
def _jax_partial_u_tw_kern(x, m, h, f1, f2):
    return qk._jax_partial_u_tw_kern(x, m, h, f1, f2)


@jjit
def _get_bounded_q_params(u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv):
    return qk._get_bounded_q_params(u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv)


@jjit
def _get_bounded_lg_drop(u_lg_drop):
    return qk._get_bounded_lg_drop(u_lg_drop)


@jjit
def _get_unbounded_lg_drop(lg_drop):
    return qk._get_unbounded_lg_drop(lg_drop)


@jjit
def _get_bounded_lg_rejuv(u_lg_rejuv, lg_drop):
    return qk._get_bounded_lg_rejuv(u_lg_rejuv, lg_drop)


@jjit
def _get_bounded_qt(u_lg_qt):
    return qk._get_bounded_qt(u_lg_qt)


@jjit
def _get_unbounded_q_params(lg_qt, lg_qs, lg_drop, lg_rejuv):
    return qk._get_unbounded_q_params(lg_qt, lg_qs, lg_drop, lg_rejuv)


_get_bounded_q_params_vmap = qk._get_bounded_q_params_vmap
_get_unbounded_q_params_vmap = qk._get_unbounded_q_params_vmap


@jjit
def _get_unbounded_qrejuv(lg_rejuv, lg_drop):
    return qk._get_unbounded_qrejuv(lg_rejuv, lg_drop)
