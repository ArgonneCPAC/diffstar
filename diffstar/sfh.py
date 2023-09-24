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
    ms_params,
    q_params,
    lgt0=LGT0,
    fb=FB,
    ms_param_type="bounded",
    q_param_type="bounded",
):
    """Calculate the star formation history of a single diffstar galaxy

    Parameters
    ----------
    tarr : ndarray, shape (n_t, )
        Age of the Universe in Gyr at which to compute the star formation history

    mah_params : ndarray, shape (4, )
        mah_params = (lgm0, logtc, early_index, late_index)

    ms_params : ndarray, shape (5, )
        By default the input ms_params will be interpreted as standard diffstar params:
            ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

        However, if ms_param_type="unbounded", then the input parameters will be
        interpreted as unbounded versions of the standard diffstar params:
            ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

        See notes for further details

    q_params : ndarray, shape (4, )
        By default the input q_params will be interpreted as standard diffstar params:
            (lg_qt, qlglgdt, lg_drop, lg_rejuv)

        However, if q_param_type="unbounded", then the input parameters will be
        interpreted as unbounded versions of the standard diffstar params:
            (u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv)

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Note that the value of this argument varies with cosmology
        Default value set by diffstar.defaults.LGT0

    fb : float, optional
        Cosmic baryon fraction
        Note that the value of this argument varies with cosmology
        Default value set by diffstar.defaults.FB

    ms_param_type : bool, optional
        Determines whether to interpret the input main sequence parameters will be
        interpreted as standard diffstar parameters ("bounded"),
        or unbounded versions of the standard params.
        Options are "unbounded" and "bounded". Default is "bounded".

    q_param_type : bool, optional
        Determines whether to interpret the input quenching parameters will be
        interpreted as standard diffstar parameters ("bounded"),
        or unbounded versions of the standard params.
        Options are "unbounded" and "bounded". Default is "bounded".

    Returns
    -------
    sfh : ndarray, shape (n_t, )
        Star formation history in units of Msun/yr

    Notes
    -----
    The ms_param_type and q_param_type options allow you call sfh_singlegal
    with input parameters that are either standard diffstar parameters or unbounded
    versions of these parameters. By default, ms_param_type and q_param_type
    are "bounded", and so the inputs will be interpreted as the standard
    diffstar parameters; note that in this case,
    it is up to the user to ensure that the numerical value of each input parameter
    lies within its physically allowed range, or else infinities and NaNs can result.

    If the param type is "unbounded", then the input parameters will instead be
    interpreted as unbounded versions of the standard diffstar parameters;
    in this case the input parameters can take on any value on the real line,
    and as an input unbounded parameter varies between (-∞, ∞),
    the corresponding diffstar parameter varies within its physically allowed range.

    """
    if ms_param_type == "bounded":
        u_ms_params = _get_unbounded_sfr_params(*ms_params)
    else:
        u_ms_params = ms_params  # interpret input ms_params as being already unbounded

    if q_param_type == "bounded":
        u_q_params = _get_unbounded_q_params(*q_params)
    else:
        u_q_params = q_params  # interpret input q_params as being already unbounded

    return _sfh_singlegal_kern(tarr, mah_params, u_ms_params, u_q_params, lgt0, fb)


@partial(jjit, static_argnames=["ms_param_type", "q_param_type"])
def sfh_galpop(
    tarr,
    mah_params,
    ms_params,
    q_params,
    lgt0=LGT0,
    fb=FB,
    ms_param_type="bounded",
    q_param_type="bounded",
):
    """Calculate the star formation history of a diffstar galaxy population

    Parameters
    ----------
    tarr : ndarray, shape (n_t, )
        Age of the Universe in Gyr at which to compute the star formation history

    mah_params : ndarray, shape (n_gals, 4)
        For each galaxy, mah_params = (lgm0, logtc, early_index, late_index)

    ms_params : ndarray, shape (n_gals, 5)
        By default the input ms_params will be interpreted as standard diffstar params:
            ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

        However, if ms_param_type="unbounded", then the input parameters will be
        interpreted as unbounded versions of the standard diffstar params:
            ms_params = (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

        See notes for further details

    q_params : ndarray, shape (n_gals, 4)
        By default the input q_params will be interpreted as standard diffstar params:
            (lg_qt, qlglgdt, lg_drop, lg_rejuv)

        However, if q_param_type="unbounded", then the input parameters will be
        interpreted as unbounded versions of the standard diffstar params:
            (u_lg_qt, u_qlglgdt, u_lg_drop, u_lg_rejuv)

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Note that the value of this argument varies with cosmology
        Default value set by diffstar.defaults.LGT0

    fb : float, optional
        Cosmic baryon fraction.
        Note that the value of this argument varies with cosmology
        Default value set by diffstar.defaults.FB

    ms_param_type : bool, optional
        Determines whether to interpret the input main sequence parameters will be
        interpreted as standard diffstar parameters ("bounded"),
        or unbounded versions of the standard params.
        Options are "unbounded" and "bounded". Default is "bounded".

    q_param_type : bool, optional
        Determines whether to interpret the input quenching parameters will be
        interpreted as standard diffstar parameters ("bounded"),
        or unbounded versions of the standard params.
        Options are "unbounded" and "bounded". Default is "bounded".

    Returns
    -------
    sfh : ndarray, shape (n_gals, n_t)
        Star formation history in units of Msun/yr

    Notes
    -----
    The ms_param_type and q_param_type options allow you call sfh_singlegal
    with input parameters that are either standard diffstar parameters or unbounded
    versions of these parameters. By default, ms_param_type and q_param_type
    are "bounded", and so the inputs will be interpreted as the standard
    diffstar parameters; note that in this case,
    it is up to the user to ensure that the numerical value of each input parameter
    lies within its physically allowed range, or else infinities and NaNs can result.

    If the param type is "unbounded", then the input parameters will instead be
    interpreted as unbounded versions of the standard diffstar parameters;
    in this case the input parameters can take on any value on the real line,
    and as an input unbounded parameter varies between (-∞, ∞),
    the corresponding diffstar parameter varies within its physically allowed range.

    """
    if ms_param_type == "bounded":
        u_ms_params = _get_unbounded_sfr_params_vmap(ms_params)
    else:
        u_ms_params = ms_params  # interpret input ms_params as being already unbounded

    if q_param_type == "bounded":
        u_q_params = _get_unbounded_q_params_vmap(q_params)
    else:
        u_q_params = q_params  # interpret input q_params as being already unbounded

    return _sfh_galpop_kern(tarr, mah_params, u_ms_params, u_q_params, lgt0, fb)
