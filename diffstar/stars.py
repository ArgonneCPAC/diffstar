"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import lax
from collections import OrderedDict
import numpy as np
from diffmah.individual_halo_assembly import _calc_halo_history
from .quenching import quenching_function
from .utils import _sigmoid, _inverse_sigmoid, _get_dt_array

FB = 0.156
TODAY = 13.8
LGT0 = jnp.log10(TODAY)
INDX_K = 9.0  # Main sequence efficiency transition speed.

DEFAULT_SFR_PARAMS = OrderedDict(
    lgmcrit=12.0, lgy_at_mcrit=-1.0, indx_lo=1.0, indx_hi=-1.0, tau_dep=2.0,
)


_SFR_PARAM_BOUNDS = OrderedDict(
    lgmcrit=(9.0, 13.5),
    lgy_at_mcrit=(-3.0, 0.0),
    indx_lo=(0.0, 5.0),
    indx_hi=(-5.0, 0.0),
    tau_dep=(0.0, 20.0),
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


SFR_PARAM_BOUNDS = calculate_sigmoid_bounds(_SFR_PARAM_BOUNDS)


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
        Diffmah halo mass accretion rate in units of Msun/yr.
    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun.
    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)
    q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes (u_qt, u_qs, u_drop, u_rejuv)
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
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar : ndarray of shape (n_times)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.

    """
    sfr = _sfr_history_from_mah(lgt, dt, dmhdt, log_mah, sfr_ms_params, q_params)
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10 ** lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


@jjit
def calculate_sm_sfr_history_from_mah(
    lgt, dt, dmhdt, log_mah, sfr_ms_params, q_params,
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
        Diffmah halo mass accretion rate in units of Msun/yr.
    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun.
    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)
    q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes (u_qt, u_qs, u_drop, u_rejuv)

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1.

    """
    sfr = _sfr_history_from_mah(lgt, dt, dmhdt, log_mah, sfr_ms_params, q_params)
    mstar = _integrate_sfr(sfr, dt)
    return mstar, sfr


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
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar : ndarray of shape (n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    dmhdt : ndarray of shape (n_times)
        Mass accretion rate in units of Msun/yr assuming h=1.
    log_mah : ndarray of shape (n_times)
        Base-10 log of cumulative peak halo mass in units of Msun assuming h=1.

    """
    dmhdt, log_mah = _calc_halo_history(lgt, *mah_params)
    mstar, sfr, fstar = calculate_sm_sfr_fstar_history_from_mah(
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
    return mstar, sfr, fstar, dmhdt, log_mah


calculate_histories_vmap = jjit(
    vmap(calculate_histories, in_axes=(None, None, 0, 0, 0, None, None, None))
)


def calculate_histories_batch(t_sim, mah_params, sfr_params, q_params, fstar_tdelay):
    """Calculate MAH and SFH histories from Diffmah and Diffstar parameters for
    large amounts of halos in small batches for memory efficiency in small systems.

    Parameters
    ----------
    t_sim : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr
    mah_params : ndarray of shape (6, )
        Best fit diffmah halo parameters. Includes
        (logt0, logmp, logtc, k, early, late)
    sfr_ms_params : ndarray of shape (5, )
        Star formation efficiency model unbounded parameters. Includes
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)
    q_params : ndarray of shape (4, )
        Quenching model unbounded parameters. Includes
        (u_qt, u_qs, u_drop, u_rejuv)
    fstar_tdelay: float
        Time interval in Gyr for fstar definition, where:
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
    mstar : ndarray of shape (n_times)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr : ndarray of shape (n_times)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar : ndarray of shape (n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    dmhdt : ndarray of shape (n_times)
        Mass accretion rate in units of Msun/yr assuming h=1.
    log_mah : ndarray of shape (n_times)
        Base-10 log of cumulative peak halo mass in units of Msun assuming h=1.

    """

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
    u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep,
):
    lgmcrit = _sigmoid(u_lgmcrit, *SFR_PARAM_BOUNDS["lgmcrit"])
    lgy_at_mcrit = _sigmoid(u_lgy_at_mcrit, *SFR_PARAM_BOUNDS["lgy_at_mcrit"])
    indx_lo = _sigmoid(u_indx_lo, *SFR_PARAM_BOUNDS["indx_lo"])
    indx_hi = _sigmoid(u_indx_hi, *SFR_PARAM_BOUNDS["indx_hi"])
    tau_dep = _sigmoid(u_tau_dep, *SFR_PARAM_BOUNDS["tau_dep"])
    bounded_params = (
        lgmcrit,
        lgy_at_mcrit,
        indx_lo,
        indx_hi,
        tau_dep,
    )
    return bounded_params


@jjit
def _get_unbounded_sfr_params(
    lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep,
):
    u_lgmcrit = _inverse_sigmoid(lgmcrit, *SFR_PARAM_BOUNDS["lgmcrit"])
    u_lgy_at_mcrit = _inverse_sigmoid(lgy_at_mcrit, *SFR_PARAM_BOUNDS["lgy_at_mcrit"])
    u_indx_lo = _inverse_sigmoid(indx_lo, *SFR_PARAM_BOUNDS["indx_lo"])
    u_indx_hi = _inverse_sigmoid(indx_hi, *SFR_PARAM_BOUNDS["indx_hi"])
    u_tau_dep = _inverse_sigmoid(tau_dep, *SFR_PARAM_BOUNDS["tau_dep"])
    bounded_params = (
        u_lgmcrit,
        u_lgy_at_mcrit,
        u_indx_lo,
        u_indx_hi,
        u_tau_dep,
    )
    return bounded_params


_get_bounded_sfr_params_vmap = jjit(vmap(_get_bounded_sfr_params, (0,) * 5, 0))
_get_unbounded_sfr_params_vmap = jjit(vmap(_get_unbounded_sfr_params, (0,) * 5, 0))


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
        Stellar mass history in Msun units.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation.
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr units for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay

    Returns
    -------
    fstar : ndarray of shape (n_times)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.

    """
    mstar_high = mstar[index_select]
    mstar_low = jax_np_interp(
        tarr[index_select] - fstar_tdelay, tarr, mstar, index_high
    )
    fstar = (mstar_high - mstar_low) / fstar_tdelay / 1e9
    return fstar


compute_fstar_vmap = jjit(vmap(compute_fstar, in_axes=(None, 0, *[None] * 3)))


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
        Diffmah halo mass accretion rate in units of Msun/yr.
    log_mah : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun.
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
    bounded_params = _get_bounded_sfr_params(*sfr_params)
    sfr_ms_params = bounded_params[:4]
    tau_dep = bounded_params[4]
    efficiency = _sfr_eff_plaw(log_mah, *sfr_ms_params)

    t_table = 10 ** lgt
    depletion_matrix = _depletion_kernel(t_table, t_table, dtarr, tau_dep)
    depletion_matrix_inst = jnp.identity(len(lgt)) / dtarr

    tau_w = jnp.where(
        tau_dep > 5.0 * jnp.mean(dtarr), jnp.ones(len(dtarr)), jnp.zeros(len(dtarr))
    )
    depletion_matrix = jnp.where(tau_w == 1, depletion_matrix, depletion_matrix_inst)

    mgas = FB * dmhdt
    integrand = mgas * depletion_matrix * dtarr
    lagged_mgas = jnp.sum(integrand, axis=1)

    lagged_sfr = lagged_mgas * efficiency

    qfrac = quenching_function(lgt, *q_params)
    sfr = qfrac * lagged_sfr
    return sfr


@jjit
def _sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi):
    """Instantaneous baryon conversion efficiency of main sequence galaxies.

    Main sequence efficiency kernel, epsilon(Mhalo).

    Parameters
    ----------
    lgm : ndarray of shape (n_times, )
        Diffmah halo mass accretion history in units of Msun.
    lgmcrit : float
        Base-10 log of the critical mass
    lgy_at_mcrit : float
        Base-10 log of the critical efficiency at critical mass.
    indx_lo : float
        Asymptotic value of the efficiency at low halo masses.
    indx_hi : float
        Asymptotic value of the efficiency at high halo masses.

    Returns
    -------
    efficiency : ndarray of shape (n_times)
        Main sequence efficiency value at each snapshot.

    """
    slope = _sigmoid(lgm, lgmcrit, INDX_K, indx_lo, indx_hi)
    eff = lgy_at_mcrit + slope * (lgm - lgmcrit)
    return 10 ** eff


@jjit
def _gas_conversion_kern(t_form, t_acc, dt, tau_dep):
    """Gas depletion kernel:
    For t_form >= t_acc -> Fdep ~ truncated triweight kernel
    For t_form < t_acc -> Fdep = 0

    Parameters
    ----------
    t_form : float
        Cosmic time in Gyr when gas is transforming into stars.
    t_acc : float
        Cosmic time in Gyr when gas accreted by the halo.
    dt : float
        Cosmic time step in Gyr between the simulated snapshot at t_form
        and the next one.
    tau_dep : float
        Cosmic time in Gyr when gas is transforming into stars.

    Returns
    -------
    tri_kern : float
        Fraction of gas accreted at time t_acc forming stars at t_form.

    """
    alpha = (tau_dep / 2.0) * (tau_dep / SFR_PARAM_BOUNDS["tau_dep"][3])
    w = (tau_dep - alpha) / 3.0
    m = t_acc + alpha

    _norm = tw_bin_jax_kern(m, w, t_acc, t_acc + tau_dep)
    _norm = 1.0 / jnp.clip(_norm, 0.01, jnp.inf)

    tri_kern = lax.cond(
        t_form < t_acc,
        lambda x: 0.0,
        lambda x: _norm * tw_bin_jax_kern(m, w, x, x + dt) / dt,
        t_form,
    )
    return tri_kern


@jjit
def tw_cuml_jax_kern(x, m, h):
    """CDF of the triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF.

    """
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 1.0,
            lambda xx: (
                -5 * xx ** 7 / 69984
                + 7 * xx ** 5 / 2592
                - 35 * xx ** 3 / 864
                + 35 * xx / 96
                + 1 / 2
            ),
            x,
        ),
        y,
    )


@jjit
def tw_bin_jax_kern(m, h, L, H):
    """Integrated bin weight for the triweight kernel.

    Parameters
    ----------
    m : array-like or scalar
        The value at which to evaluate the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.
    L : array-like or scalar
        The lower bin limit.
    H : array-like or scalar
        The upper bin limit.

    Returns
    -------
    bin_prob : array-like or scalar
        The value of the kernel integrated over the bin.

    """
    return tw_cuml_jax_kern(H, m, h) - tw_cuml_jax_kern(L, m, h)


_a, _b = (0, None, 0, None), (None, 0, None, None)
_depletion_kernel = jjit(vmap(vmap(_gas_conversion_kern, in_axes=_b), in_axes=_a))


@jjit
def jax_np_interp(x, xt, yt, indx_hi):
    """JAX-friendly implementation of np.interp.
    Requires indx_hi to be precomputed, e.g., using np.searchsorted.

    Parameters
    ----------
    x : ndarray of shape (n, )
        Abscissa values in the interpolation
    xt : ndarray of shape (k, )
        Lookup table for the abscissa
    yt : ndarray of shape (k, )
        Lookup table for the ordinates

    Returns
    -------
    y : ndarray of shape (n, )
        Result of linear interpolation

    """
    indx_lo = indx_hi - 1
    xt_lo = xt[indx_lo]
    xt_hi = xt[indx_hi]
    dx_tot = xt_hi - xt_lo
    yt_lo = yt[indx_lo]
    yt_hi = yt[indx_hi]
    dy_tot = yt_hi - yt_lo
    m = dy_tot / dx_tot
    y = yt_lo + m * (x - xt_lo)
    return y


def fstar_tools(t_sim, fstar_tdelay=1.0):
    """Calculate the snapshots used by fstar.

    Parameters
    ----------
        t_sim: ndarray of shape (nt, )
            Cosmic time of each simulated snapshot in Gyr.
        fstar_tdelay: float
            Time interval in Gyr for fstar definition.
            fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]

    Returns
    -------
        index_select: ndarray of shape (n_times_fstar, )
            Snapshot indices used in fstar computation.
        fstar_indx_high: ndarray of shape (n_times_fstar, )
            Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]

    """
    fstar_indx_high = np.searchsorted(t_sim, t_sim - fstar_tdelay)
    _mask = t_sim > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_sim))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]
    return index_select, fstar_indx_high
