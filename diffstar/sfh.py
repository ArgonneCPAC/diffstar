"""
"""
from jax import jit as jjit

from .defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from .kernel_builders import get_sfh_from_mah_kern

_sfh_singlegal_kern = get_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    lgt0=LGT0,
    tacc_integration_min=T_BIRTH_MIN,
    fb=FB,
    tobs_loop="scan",
)
_sfh_galpop_kern = get_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    lgt0=LGT0,
    tacc_integration_min=T_BIRTH_MIN,
    fb=FB,
    galpop_loop="vmap",
    tobs_loop="scan",
)


@jjit
def sfh_singlegal(tarr, mah_params, u_ms_params, u_q_params):
    """Calculate the star formation history of a single diffstar galaxy

    Parameters
    ----------
    tarr : ndarray, shape (n_t, )

    mah_params : ndarray, shape (4, )
        mah_params = (lgm0, logtc, early_index, late_index)

    u_ms_params : ndarray, shape (5, )
        u_ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray, shape (4, )
        u_q_params = (u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv)

    Returns
    -------
    sfh : ndarray, shape (n_t, )

    """
    return _sfh_singlegal_kern(tarr, mah_params, u_ms_params, u_q_params)


@jjit
def sfh_galpop(tarr, mah_params, u_ms_params, u_q_params):
    """Calculate the star formation history of a diffstar galaxy population

    Parameters
    ----------
    tarr : ndarray, shape (n_gals, n_t)

    mah_params : ndarray, shape (n_gals, 4)
        For each galaxy, mah_params = (lgm0, logtc, early_index, late_index)

    u_ms_params : ndarray, shape (n_gals, 5)
        For each galaxy,
        u_ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray, shape (n_gals, 4)
        For each galaxy, u_q_params = (u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv)


    Returns
    -------
    sfh : ndarray, shape (n_gals, n_t)

    """
    return _sfh_galpop_kern(tarr, mah_params, u_ms_params, u_q_params)
