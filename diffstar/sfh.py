"""
"""
from functools import partial

from jax import jit as jjit

from .defaults import DEFAULT_N_STEPS, FB, LGT0, T_BIRTH_MIN
from .kernels.kernel_builders import get_sfh_from_mah_kern
from .kernels.main_sequence_kernels import (
    _get_unbounded_sfr_params,
    _get_unbounded_sfr_params_vmap,
)
from .kernels.quenching_kernels import (
    _get_unbounded_q_params,
    _get_unbounded_q_params_vmap,
)

_sfh_singlegal_kern = get_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    tobs_loop="scan",
)
_sfh_galpop_kern = get_sfh_from_mah_kern(
    n_steps=DEFAULT_N_STEPS,
    tacc_integration_min=T_BIRTH_MIN,
    galpop_loop="vmap",
    tobs_loop="scan",
)


@partial(jjit, static_argnames=["ms_param_type", "q_param_type"])
def sfh_singlegal(
    tarr,
    mah_params,
    u_ms_params,
    u_q_params,
    lgt0=LGT0,
    fb=FB,
    ms_param_type="unbounded",
    q_param_type="unbounded",
):
    """Calculate the star formation history of a single diffstar galaxy

    Parameters
    ----------
    tarr : ndarray, shape (n_t, )

    mah_params : ndarray, shape (4, )
        mah_params = (lgm0, logtc, early_index, late_index)

    u_ms_params : ndarray, shape (5, )
        By default the input u_ms_params will be interpreted as the
        unbounded versions of the standard diffstar params:
            u_ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

        However, if ms_param_type="bounded", then the input parameters parameters
        will be interpreted as the standard diffstar params:
            u_ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

        See notes for further details

    u_q_params : ndarray, shape (4, )
        u_q_params = (u_lg_qt, u_lg_lg_q_dt, u_lg_drop, u_lg_rejuv)

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Default value set by diffstar.defaults.LGT0

    fb : float, optional
        Cosmic baryon fraction
        Default value set by diffstar.defaults.FB

    ms_param_type : bool, optional
        Determines whether to interpret the input main sequence parameters as being
        unbounded version of the diffstar params. Default is "unbounded".

    q_param_type : bool, optional
        Determines whether to interpret the input quenching parameters as being
        unbounded version of the diffstar params. Default is "unbounded".

    Returns
    -------
    sfh : ndarray, shape (n_t, )

    Notes
    -----
    The ms_param_type and q_param_type options allow you call sfh_singlegal
    with input parameters that are either standard bounded parameters or unbounded ones.
    By default, ms_param_type and q_param_type are "unbounded",
    and so the input parameters can take on any value on the real line,
    and as an input unbounded parameter varies between (-∞, ∞),
    the corresponding diffstar parameter varies within its physically allowed range.
    But if the param type is "bounded", then the input parameters will instead be
    interpreted as the standard diffstar parameters; note that in this case,
    the numerical value of each input parameters should line within the
    physically allowed range, or else infinities and NaNs can result.

    """
    if ms_param_type == "bounded":
        u_ms_params = _get_unbounded_sfr_params(*u_ms_params)
    if q_param_type == "bounded":
        u_q_params = _get_unbounded_q_params(*u_q_params)
    return _sfh_singlegal_kern(tarr, mah_params, u_ms_params, u_q_params, lgt0, fb)


@partial(jjit, static_argnames=["ms_param_type", "q_param_type"])
def sfh_galpop(
    tarr,
    mah_params,
    u_ms_params,
    u_q_params,
    lgt0=LGT0,
    fb=FB,
    ms_param_type="unbounded",
    q_param_type="unbounded",
):
    """Calculate the star formation history of a diffstar galaxy population

    Parameters
    ----------
    tarr : ndarray, shape (n_gals, n_t)

    mah_params : ndarray, shape (n_gals, 4)
        For each galaxy, mah_params = (lgm0, logtc, early_index, late_index)

    u_ms_params : ndarray, shape (n_gals, 5)
        For each galaxy, by default the input u_ms_params will be interpreted as the
        unbounded versions of the standard diffstar params:
            u_ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

        However, if ms_param_type="bounded", then the input parameters parameters
        will be interpreted as the standard diffstar params:
            u_ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

        See notes for further details

    u_q_params : ndarray, shape (n_gals, 4)
        For each galaxy, by default the input u_q_params will be interpreted as the
        unbounded versions of the standard diffstar params:
            u_q_params = (u_lg_qt, u_lg_lg_q_dt, u_lg_drop, u_lg_rejuv)

        However, if q_param_type="bounded", then the input parameters parameters
        will be interpreted as the standard diffstar params:
            u_q_params = (lg_qt, lg_lg_q_dt, lg_drop, lg_rejuv)

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Default value set by diffstar.defaults.LGT0

    fb : float, optional
        Cosmic baryon fraction
        Default value set by diffstar.defaults.FB

    ms_param_type : bool, optional
        Determines whether to interpret the input main sequence parameters as being
        unbounded version of the diffstar params. Default is "unbounded".

    q_param_type : bool, optional
        Determines whether to interpret the input quenching parameters as being
        unbounded version of the diffstar params. Default is "unbounded".

    Returns
    -------
    sfh : ndarray, shape (n_gals, n_t)

    Notes
    -----
    The ms_param_type and q_param_type options allow you call sfh_singlegal
    with input parameters that are either standard bounded parameters or unbounded ones.
    By default, ms_param_type and q_param_type are "unbounded",
    and so the input parameters can take on any value on the real line,
    and as an input unbounded parameter varies between (-∞, ∞),
    the corresponding diffstar parameter varies within its physically allowed range.
    But if the param type is "bounded", then the input parameters will instead be
    interpreted as the standard diffstar parameters; note that in this case,
    the numerical value of each input parameters should line within the
    physically allowed range, or else infinities and NaNs can result.

    """
    if ms_param_type == "bounded":
        u_ms_params = _get_unbounded_sfr_params_vmap(u_ms_params)
    if q_param_type == "bounded":
        u_q_params = _get_unbounded_q_params_vmap(u_q_params)
    return _sfh_galpop_kern(tarr, mah_params, u_ms_params, u_q_params, lgt0, fb)
