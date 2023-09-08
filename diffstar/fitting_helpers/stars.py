"""
"""
import numpy as np
from jax import jit as jjit

from ..kernels import main_sequence_kernels as msk
from ..kernels import sfr_kernels as sfrk
from ..utils import _get_dt_array


@jjit
def calculate_sm_sfr_fstar_history_from_mah(
    lgt,
    dt,
    dmhdt,
    log_mah,
    sfr_ms_params,
    q_params,
    index_select,
    index_high,
    fstar_tdelay,
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

    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    q_params : ndarray of shape (4, )
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
    return sfrk.calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        sfr_ms_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )


@jjit
def calculate_sm_sfr_history_from_mah(
    lgt,
    dt,
    dmhdt,
    log_mah,
    sfr_ms_params,
    q_params,
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

    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes (u_qt, u_qs, u_drop, u_rejuv)

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1

    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1

    """
    return sfrk.calculate_sm_sfr_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        sfr_ms_params,
        q_params,
    )


@jjit
def calculate_histories(
    lgt,
    dt,
    mah_params,
    sfr_ms_params,
    q_params,
    index_select,
    index_high,
    fstar_tdelay,
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

    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    q_params : ndarray of shape (4, )
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
    return sfrk.calculate_histories(
        lgt,
        dt,
        mah_params,
        sfr_ms_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )


calculate_histories_vmap = sfrk.calculate_histories_vmap


def calculate_histories_batch(t_sim, mah_params, sfr_params, q_params, fstar_tdelay):
    """Calculate MAH and SFH histories for a large population of halos.


    Parameters
    ----------
    t_sim : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr

    mah_params : ndarray of shape (ng, 6)
        Best fit diffmah halo parameters. Includes
        (logt0, logmp, logtc, k, early, late)

    sfr_params : ndarray of shape (ng, 5)
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    q_params : ndarray of shape (ng, 4)
        Quenching model unbounded parameters. Includes
        (u_qt, u_qs, u_drop, u_rejuv)

    fstar_tdelay: float
        Time interval in Gyr for fstar definition, where:
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
    mstar : ndarray of shape (ng, n_times)
        Cumulative stellar mass history in units of Msun assuming h=1

    sfr : ndarray of shape (ng, n_times)
        Star formation rate history in units of Msun/yr assuming h=1

    fstar : ndarray of shape (ng, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1

    dmhdt : ndarray of shape (ng, n_times)
        Mass accretion rate in units of Msun/yr assuming h=1

    log_mah : ndarray of shape (ng, n_times)
        Base-10 log of cumulative peak halo mass in units of Msun assuming h=1

    Notes
    -----
    Histories are calculated from Diffmah and Diffstar parameters in small batches
    for memory efficiency in small systems.

    The input tsim array is in units of Gyr and should be strictly monotonically
    increasing. The tsim array should begin at roughly t[0]~0.1, and have spacing
    at least as fine as dt~0.25. The input tsim array is used to integrate SFR to
    compute Mstar, and so tsim should be finely-spaced enough for the desired
    accuracy of the integration.

    Note that mstar[0] = sfr[0] * (t_sim[1] - t_sim[0]) * 1e9, and so by definition
    sSFR[0] = sfr[0] / mstar[0] = 1 / (t_sim[1] - t_sim[0]) / 1e9.
    """
    assert np.all(
        np.diff(t_sim) > 0.0
    ), "t_sim needs to be strictly monotonically increasing"
    assert np.all(t_sim > 0.0), "t_sim needs to be strictly positive"
    _msg = "Diffmah and Diffstar parameters need to have the same number of galaxies"
    assert len(mah_params) == len(sfr_params) == len(q_params), _msg

    dt = _get_dt_array(t_sim)
    lgt = np.log10(t_sim)
    index_select, index_high = fstar_tools(t_sim, fstar_tdelay=fstar_tdelay)

    ng = len(mah_params)
    nt = len(lgt)
    nt2 = len(index_high)
    indices = np.array_split(np.arange(ng), max(int(ng / 5000), 1))

    mstar = np.zeros((ng, nt))
    sfr = np.zeros((ng, nt))
    fstar = np.zeros((ng, nt2))
    dmhdt = np.zeros((ng, nt))
    log_mah = np.zeros((ng, nt))

    for inds in indices:
        _res = calculate_histories_vmap(
            lgt,
            dt,
            mah_params[inds],
            sfr_params[inds],
            q_params[inds],
            index_select,
            index_high,
            fstar_tdelay,
        )
        mstar[inds] = _res[0]
        sfr[inds] = _res[1]
        fstar[inds] = _res[2]
        dmhdt[inds] = _res[3]
        log_mah[inds] = _res[4]

    return mstar, sfr, fstar, dmhdt, log_mah


@jjit
def _get_bounded_sfr_params(
    u_lgmcrit,
    u_lgy_at_mcrit,
    u_indx_lo,
    u_indx_hi,
    u_tau_dep,
):
    return msk._get_bounded_sfr_params(
        u_lgmcrit,
        u_lgy_at_mcrit,
        u_indx_lo,
        u_indx_hi,
        u_tau_dep,
    )


@jjit
def _get_unbounded_sfr_params(
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,
    tau_dep,
):
    return msk._get_unbounded_sfr_params(
        lgmcrit,
        lgy_at_mcrit,
        indx_lo,
        indx_hi,
        tau_dep,
    )


_get_bounded_sfr_params_vmap = msk._get_bounded_sfr_params_vmap
_get_unbounded_sfr_params_vmap = msk._get_unbounded_sfr_params_vmap


@jjit
def _integrate_sfr(sfr, dt):
    """Calculate the cumulative stellar mass history."""
    return sfrk._integrate_sfr(sfr, dt)


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
    return sfrk.compute_fstar(tarr, mstar, index_select, index_high, fstar_tdelay)


compute_fstar_vmap = sfrk.compute_fstar_vmap


@jjit
def _sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params, q_params):
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
    return sfrk._sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params, q_params)


@jjit
def _ms_sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params):
    """Main Sequence formation history of an individual galaxy."""

    return sfrk._ms_sfr_history_from_mah(lgt, dtarr, dmhdt, log_mah, sfr_params)


@jjit
def _sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi):
    """Instantaneous baryon conversion efficiency of main sequence galaxies

    Main sequence efficiency kernel, epsilon(Mhalo)

    Parameters
    ----------
    lgm : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun

    lgmcrit : float
        Base-10 log of the critical mass
    lgy_at_mcrit : float
        Base-10 log of the critical efficiency at critical mass

    indx_lo : float
        Asymptotic value of the efficiency at low halo masses

    indx_hi : float
        Asymptotic value of the efficiency at high halo masses

    Returns
    -------
    efficiency : ndarray of shape (n_times)
        Main sequence efficiency value at each snapshot

    """
    return sfrk._sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi)


def fstar_tools(t_sim, fstar_tdelay=1.0):
    """Calculate the snapshots used by fstar.

    Parameters
    ----------
        t_sim: ndarray of shape (nt, )
            Cosmic time of each simulated snapshot in Gyr

        fstar_tdelay: float
            Time interval in Gyr for fstar definition.
            fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
        index_select: ndarray of shape (n_times_fstar, )
            Snapshot indices used in fstar computation

        fstar_indx_high: ndarray of shape (n_times_fstar, )
            Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]

    """
    fstar_indx_high = np.searchsorted(t_sim, t_sim - fstar_tdelay)
    _mask = t_sim > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_sim))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]
    return index_select, fstar_indx_high
