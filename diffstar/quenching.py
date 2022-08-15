"""Model for the quenching of individual galaxies."""
from collections import OrderedDict
from jax import jit as jjit
from .kernels.quenching_kernels import _get_bounded_q_params, _quenching_kern

DEFAULT_Q_PARAMS = OrderedDict(
    u_lg_qt=1.0, u_lg_qs=-0.3, u_lg_drop=-1.0, u_lg_rejuv=-0.5
)


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
