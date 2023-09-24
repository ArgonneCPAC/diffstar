"""
"""
from diffmah.individual_halo_assembly import _calc_halo_history
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..defaults import FB
from ..kernels.gas_consumption import _get_lagged_gas
from ..kernels.main_sequence_kernels import (
    MS_BOUNDING_SIGMOID_PDICT,
    _get_bounded_sfr_params,
    _sfr_eff_plaw,
)
from ..kernels.quenching_kernels import _quenching_kern_u_params
from ..utils import jax_np_interp


@jjit
def calculate_sm_sfr_fstar_history_from_mah(
    lgt,
    dt,
    dmhdt,
    log_mah,
    u_ms_params,
    u_q_params,
    index_select,
    index_high,
    fstar_tdelay,
    fb=FB,
):
    """Calculate individual galaxy SFH from precalculated halo MAH

    The accretion rate of gas is proportional to
    the accretion rate of the halo; main sequence galaxies transform accreted
    gas into stars over a time depletion timescale with an efficiency that
    depends on the instantaneous halo mass; some galaxies experience a quenching
    event and may subsequently experience a rejuvenated star formation.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Base-10 log of cosmic time of each simulated snapshot in Gyr

    dt : ndarray of shape (n_times, )
        Cosmic time steps between each simulated snapshot in Gyr

    dmhdt : ndarray of shape (n_times, )
        Diffmah halo mass accretion rate in units of Msun/yr

    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun

    u_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes (u_qt, u_qs, u_drop, u_rejuv)

    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation

    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]

    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1

    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1

    fstar : ndarray of shape (n_times)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1

    """
    sfr = _sfr_history_from_mah(lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params, fb=fb)
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10**lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


@jjit
def calculate_sm_sfr_history_from_mah(
    lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params, fb=FB
):
    """Calculate individual galaxy SFH from precalculated halo MAH

    The accretion rate of gas is proportional to
    the accretion rate of the halo; main sequence galaxies transform accreted
    gas into stars over a time depletion timescale with an efficiency that
    depends on the instantaneous halo mass; some galaxies experience a quenching
    event and may subsequently experience a rejuvenated star formation.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Base-10 log of cosmic time of each simulated snapshot in Gyr

    dt : ndarray of shape (n_times, )
        Cosmic time steps between each simulated snapshot in Gyr

    dmhdt : ndarray of shape (n_times, )
        Diffmah halo mass accretion rate in units of Msun/yr

    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun

    u_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes (u_qt, u_qs, u_drop, u_rejuv)

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1

    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1

    """
    sfr = _sfr_history_from_mah(lgt, dt, dmhdt, log_mah, u_ms_params, u_q_params, fb=fb)
    mstar = _integrate_sfr(sfr, dt)
    return mstar, sfr


@jjit
def calculate_histories(
    lgt,
    dt,
    mah_params,
    u_ms_params,
    u_q_params,
    index_select,
    index_high,
    fstar_tdelay,
    fb=FB,
):
    """Calculate individual halo mass MAH and galaxy SFH

    The accretion rate of gas is proportional to the accretion rate of the halo;
    main sequence galaxies transform accreted
    gas into stars over a time depletion timescale with an efficiency that
    depends on the instantaneous halo mass; some galaxies experience a quenching
    event and may subsequently experience a rejuvenated star formation.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Base-10 log of cosmic time of each simulated snapshot in Gyr

    dt : ndarray of shape (n_times, )
        Cosmic time steps between each simulated snapshot in Gyr

    mah_params : ndarray of shape (6, )
        Best fit diffmah halo parameters. Includes
        (logt0, logmp, logtc, k, early, late)

    u_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes
        (u_qt, u_qs, u_drop, u_rejuv)

    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation.

    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]

    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1

    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1

    fstar : ndarray of shape (n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1

    dmhdt : ndarray of shape (n_times)
        Mass accretion rate in units of Msun/yr assuming h=1

    log_mah : ndarray of shape (n_times)
        Base-10 log of cumulative peak halo mass in units of Msun assuming h=1

    """
    dmhdt, log_mah = _calc_halo_history(lgt, *mah_params)
    mstar, sfr, fstar = calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        u_ms_params,
        u_q_params,
        index_select,
        index_high,
        fstar_tdelay,
        fb=fb,
    )
    return mstar, sfr, fstar, dmhdt, log_mah


calculate_histories_vmap = jjit(
    vmap(calculate_histories, in_axes=(None, None, 0, 0, 0, None, None, None, None))
)


@jjit
def _integrate_sfr(sfr, dt):
    """Calculate the cumulative stellar mass history."""
    return jnp.cumsum(sfr * dt) * 1e9


@jjit
def compute_fstar(tarr, mstar, index_select, index_high, fstar_tdelay):
    """Time averaged SFH that has ocurred over some previous time period

    fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay

    Parameters
    ----------
    tarr : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    mstar : ndarray of shape (n_times, )
        Stellar mass history in Msun units

    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation

    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]

    fstar_tdelay: float
        Time interval in Gyr units for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay

    Returns
    -------
    fstar : ndarray of shape (n_times)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1

    """
    mstar_high = mstar[index_select]
    mstar_low = jax_np_interp(
        tarr[index_select] - fstar_tdelay, tarr, mstar, index_high
    )
    fstar = (mstar_high - mstar_low) / fstar_tdelay / 1e9
    return fstar


compute_fstar_vmap = jjit(vmap(compute_fstar, in_axes=(None, 0, *[None] * 3)))


@jjit
def _sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params, q_params, fb=FB):
    """Star formation history of an individual galaxy.

    SFH(t) = Quenching(t) x epsilon(Mhalo) int Depletion(t|t') x Mgas(t') dt'.
    Mgas(t) = FB x dMhalo(t)/dt

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Base-10 log of cosmic time of each simulated snapshot in Gyr

    dtarr : ndarray of shape (n_times, )
        Cosmic time steps between each simulated snapshot in Gyr

    dmhdt : ndarray of shape (n_times, )
        Diffmah halo mass accretion rate in units of Msun/yr

    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun

    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes
        (u_qt, u_qs, u_drop, u_rejuv)

    Returns
    -------
    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1.

    """
    ms_sfr = _ms_sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params, fb=fb)
    qfrac = _quenching_kern_u_params(lgt, *q_params)
    sfr = qfrac * ms_sfr
    return sfr


@jjit
def _ms_sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, u_ms_params, fb=FB):
    """Main Sequence formation history of an individual galaxy."""

    ms_params = _get_bounded_sfr_params(*u_ms_params)
    sfr_ms_params = ms_params[:4]
    tau_dep = ms_params[4]
    efficiency = _sfr_eff_plaw(log_mah, *sfr_ms_params)

    tau_dep_max = MS_BOUNDING_SIGMOID_PDICT["tau_dep"][3]
    lagged_mgas = _get_lagged_gas(lgt, dtarr, dmhdt, tau_dep, tau_dep_max, fb)

    ms_sfr = lagged_mgas * efficiency
    return ms_sfr
