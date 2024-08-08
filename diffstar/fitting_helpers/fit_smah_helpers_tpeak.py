"""
"""
import os
import warnings

import h5py
import numpy as np

from jax import grad
from jax import jit as jjit
from jax import numpy as jnp

from diffmah.diffmah_kernels import calculate_sm_sfr_fstar_history_from_mah, _diffmah_kern
from diffmah.defaults import LGT0

from ..defaults import (
    DEFAULT_MS_PDICT,
    DEFAULT_Q_PDICT,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
)
from ..kernels.main_sequence_kernels_tpeak import (
    _get_bounded_sfr_params_vmap,
    _get_unbounded_sfr_params,
)
from ..kernels.quenching_kernels import (
    _get_bounded_lg_drop,
    _get_bounded_q_params_vmap,
    _get_bounded_qt,
    _get_unbounded_q_params,
    _get_unbounded_qrejuv,
)
from .fitting_kernels import compute_fstar

from ..utils import _sigmoid

T_FIT_MIN = 1.0  # Only fit snapshots above this threshold. Gyr units.
DLOGM_CUT = 3.5  # Only fit SMH within this dex of the present day stellar mass.
MIN_MASS_CUT = 7.0  # Only fit SMH above this threshold. Log10(Msun) units.
FSTAR_TIME_DELAY = 1.0  # Time period of averaged SFH (aka fstar). Gyr units.
SSFRH_FLOOR = 1e-12  # Clip SFH to this minimum sSFR value. 1/yr units.


@jjit
def loss_default(params, loss_data):
    """
    MSE loss function for fitting individual stellar mass histories.
    The parameters k, indx_hi are fixed.

    """
    (
        lgt,
        dt,
        dmhdt,
        log_mah,
        sm_target,
        log_sm_target,
        sfr_target,
        fstar_target,
        index_select,
        fstar_indx_high,
        fstar_tdelay,
        ssfrh_floor,
        weight,
        weight_fstar,
        t_fstar_max,
        fixed_hi,
    ) = loss_data

    sfr_params = [*params[0:3], fixed_hi, params[3]]
    q_params = params[4:8]

    _res = calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        sfr_params,
        q_params,
        index_select,
        fstar_indx_high,
        fstar_tdelay,
    )

    mstar, sfr, fstar = _res
    mstar = jnp.log10(mstar)
    fstar = jnp.log10(fstar)

    sfr_res = 1e8 * (sfr - sfr_target) / sm_target
    sfr_res = jnp.clip(sfr_res, -1.0, 1.0)

    loss = jnp.mean(((mstar - log_sm_target) / weight) ** 2)
    loss += jnp.mean(((fstar - fstar_target) / weight_fstar) ** 2)
    loss += jnp.mean((sfr_res / weight) ** 2)

    qt = _get_bounded_qt(q_params[0])
    loss += _sigmoid(qt - t_fstar_max, 0.0, 50.0, 100.0, 0.0)
    return loss


loss_grad_default = jjit(grad(loss_default, argnums=(0)))


def loss_grad_default_np(params, data):
    return np.array(loss_grad_default(params, data)).astype(float)

def get_loss_data_default(
    t_sim,
    dt,
    sfrh,
    log_smah_sim,
    logmp,
    mah_params,
    t_peak,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    mass_fit_min=MIN_MASS_CUT,
    fstar_tdelay=FSTAR_TIME_DELAY,
    ssfrh_floor=SSFRH_FLOOR,
    lgt0=LGT0,
):
    """Retrieve the target data passed to the optimizer when fitting the halo
    SFH model for the case in which the parameters k, indx_hi are fixed.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr units.
    dt : ndarray of shape (nt, )
        Cosmic time steps between each simulated snapshot in Gyr units.
    sfrh : ndarray of shape (nt, )
        Star formation history of simulated snapshots in Msun/yr units.
    log_smah_sim : ndarray of shape (nt, )
        Base-10 log of cumulative stellar mass in Msun units.
    logmp : float
        Base-10 log present day halo mass in Msun units.
    mah_params : ndarray of shape (4, )
        Best fit diffmah halo parameters. Includes (logtc, k, early, late).
    dlogm_cut : float, optional
        Additional quantity used to place a cut on which simulated snapshots
        are used to define the target halo SFH.
        Snapshots will not be used when log_smah_sim falls below
        log_smah_sim[-1] - dlogm_cut.
        Default is set as global at top of module.
    t_fit_min : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo SFH. The value of t_fit_min defines the minimum cosmic
        time in Gyr used to define the target SFH.
        Default is set as global at top of module.
    mass_fit_min : float
        Quantity used to place a cut on which simulated snapshots are used to
        define the target halo SFH.
        The value mass_fit_min is the base-10 log of the minimum stellar mass in the SFH
        used as target data. The final mass_fit_min cut is equal to
        min(log_smah_sim[-1] - 0.5, mass_fit_min).
        Default is set as global at top of module.
    fstar_tdelay : float
        Time interval in Gyr for fstar definition.
        fstar = mstar(t) - mstar(t-fstar_tdelay)
        Default is set as global at top of module.
    ssfrh_floor : float
        Lower bound value of star formation history used in the fits.
        SFH(t) = max(SFH(t), SMH(t) * ssfrh_floor)
        Default is set as global at top of module.

    Returns
    -------
    p_init : ndarray of shape (5, )
        Initial guess at the unbounded value of the best-fit parameter.
        Here we have p_init = (u_lgm, u_lgy, u_l, u_h, u_dt)
    loss_data : sequence consisting of the following data
        logt: ndarray of shape (nt, )
            Base-10 log of cosmic time of each simulated snapshot in Gyr.
        dt : ndarray of shape (nt, )
            Cosmic time steps between each simulated snapshot in Gyr
        dmhdt : ndarray of shape (nt, )
            Diffmah halo mass accretion rate in units of Msun/yr.
        log_mah : ndarray of shape (nt, )
            Diffmah halo mass accretion history in units of Msun.
        smh : ndarray of shape (nt, )
            Cumulative stellar mass history in Msun.
        log_smah_sim : ndarray of shape (nt, )
            Base-10 log of cumulative stellar mass in Msun.
        sfrh : ndarray of shape (nt, )
            Star formation history in Msun/yr.
        log_fstar_sim : ndarray of shape (nt, )
            Base-10 log of cumulative SFH averaged over a timescale in Msun/yr.
        index_select: ndarray of shape (n_times_fstar, )
            Snapshot indices used in fstar computation.
        fstar_indx_high: ndarray of shape (n_times_fstar, )
            Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
        fstar_tdelay: float
            Time interval in Gyr for fstar definition.
            fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
        ssfrh_floor : float
            Lower bound value of star formation history used in the fits.
        weight : ndarray of shape (nt, )
            Weight for each snapshot, to effectively remove from the fit
            the SMH snapshots that fall below the threshold mass.
        weight_fstar : ndarray of shape (n_times_fstar, )
            Weight for each snapshot, to effectively remove from the fit
            the SFH snapshots that fall below the threshold mass.
        t_fstar_max : float
            Base-10 log of the cosmic time where SFH target history peaks.
        fixed_hi : float
            Fixed value of the unbounded diffstar parameter indx_hi

    """
    fstar_indx_high = np.searchsorted(t_sim, t_sim - fstar_tdelay)
    _mask = t_sim > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_sim))[_mask]
    fstar_indx_high = fstar_indx_high[_mask]

    smh = 10**log_smah_sim

    fstar_sim = compute_fstar(t_sim, smh, index_select, fstar_indx_high, fstar_tdelay)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssfrh = fstar_sim / smh[index_select]
        ssfrh = np.clip(ssfrh, ssfrh_floor, np.inf)
        fstar_sim = ssfrh * smh[index_select]
        log_fstar_sim = np.where(
            fstar_sim == 0.0, np.log10(fstar_sim.max()) - 3.0, np.log10(fstar_sim)
        )

    logt = jnp.log10(t_sim)
    logtmp = np.log10(t_sim[-1])
    dmhdt, log_mah = _diffmah_kern(mah_params, t_sim, t_peak, lgt0)

    weight, weight_fstar = get_weights(
        t_sim,
        log_smah_sim,
        log_fstar_sim,
        fstar_indx_high,
        dlogm_cut,
        t_fit_min,
        mass_fit_min,
    )

    t_fstar_max = logt[index_select][np.argmax(log_fstar_sim)]

    default_sfr_params = np.array(DEFAULT_U_MS_PARAMS)
    default_sfr_params[0] = np.clip(0.3 * (logmp - 11.0) + 11.4, 11.0, 13.0)
    default_sfr_params[1] = np.clip(0.2 * (logmp - 11.0) - 0.7, -1.5, -0.2)
    default_sfr_params[2] = np.clip(0.7 * (logmp - 11.0) - 0.3, 0.2, 3.0)
    default_sfr_params[4] = np.clip(-8.0 * (logmp - 11.0) + 15, 2.0, 15.0)
    u_default_sfr_params = np.array(_get_unbounded_sfr_params(*default_sfr_params))

    sfr_ms_params = np.zeros(4)
    sfr_ms_params[0:3] = u_default_sfr_params[0:3]
    sfr_ms_params[3] = u_default_sfr_params[4]
    fixed_hi = u_default_sfr_params[3]

    sfr_ms_params_err = np.array([0.5, 0.5, 1.0, 3.0])

    default_q_params = np.array(DEFAULT_U_Q_PARAMS)
    default_q_params[0] = np.clip(-0.5 * (logmp - 11.0) + 1.5, 0.7, 1.5)
    default_q_params[2] = -2.0
    q_params = np.array(_get_unbounded_q_params(*default_q_params))
    q_params_err = np.array([0.3, 0.5, 0.3, 0.3])

    loss_data = (
        logt,
        dt,
        dmhdt,
        log_mah,
        smh,
        log_smah_sim,
        sfrh,
        log_fstar_sim,
        index_select,
        fstar_indx_high,
        fstar_tdelay,
        ssfrh_floor,
        weight,
        weight_fstar,
        t_fstar_max,
        fixed_hi,
    )
    p_init = (
        np.concatenate((sfr_ms_params, q_params)),
        np.concatenate((sfr_ms_params_err, q_params_err)),
    )
    return p_init, loss_data


def get_outline_default(halo_id, loss_data, p_best, loss_best, success):
    """Return the string storing fitting results that will be written to disk"""
    fixed_hi = loss_data[-1]
    sfr_params = np.zeros(5)
    sfr_params[0:3] = p_best[0:3]
    sfr_params[3] = fixed_hi
    sfr_params[4] = p_best[3]
    q_params = p_best[4:8]
    _d = np.concatenate((sfr_params, q_params)).astype("f4")
    data_out = (*_d, float(loss_best))
    out = str(halo_id) + " " + " ".join(["{:.5e}".format(x) for x in data_out])
    out = out + " " + str(success)
    return out + "\n"


def get_weights(
    t_sim,
    log_smah_sim,
    log_fstar_sim,
    fstar_indx_high,
    dlogm_cut,
    t_fit_min,
    mass_fit_min,
):
    """Calculate weights to mask target SMH and fstar target data.

    Parameters
    ----------
    t_sim : ndarray of shape (nt, )
        Cosmic time of each simulated snapshot in Gyr units.
    log_smah_sim : ndarray of shape (nt, )
        Base-10 log of cumulative stellar mass in Msun units.
    log_fstar_sim : ndarray of shape (nt, )
        Base-10 log of SFH averaged over a time period in Msun/yr units.
    fstar_indx_high: ndarray of shape (n_times_fstar, )
        Indices from np.searchsorted(t, t - fstar_tdelay)[index_select]
    dlogm_cut : float, optional
        Additional quantity used to place a cut on which simulated snapshots
        are used to define the target halo SFH.
        Snapshots will not be used when log_smah_sim falls below
        log_smah_sim[-1] - dlogm_cut.
    t_fit_min : float, optional
        Additional quantity used to place a cut on which simulated snapshots are used to
        define the target halo SFH. The value of t_fit_min defines the minimum cosmic
        time in Gyr used to define the target SFH.
    mass_fit_min : float
        Quantity used to place a cut on which simulated snapshots are used to
        define the target halo SFH.
        The value mass_fit_min is the base-10 log of the minimum stellar mass in the SFH
        used as target data. The final mass_fit_min cut is equal to
        min(log_smah_sim[-1] - 0.5, mass_fit_min).

    Returns
    -------
    weight : ndarray of shape (nt, )
        Weight for each snapshot, to effectively remove from the fit
        the SMH snapshots that fall below the threshold mass.
    weight_fstar : ndarray of shape (n_times_fstar, )
        Weight for each snapshot, to effectively remove from the fit
        the SFH snapshots that fall below the threshold mass.

    """
    mass_fit_min = min(log_smah_sim[-1] - 0.5, mass_fit_min)

    mask = log_smah_sim > (log_smah_sim[-1] - dlogm_cut)
    mask &= log_smah_sim > mass_fit_min
    mask &= t_sim >= t_fit_min

    weight = np.ones_like(t_sim)
    weight[~mask] = 1e10
    weight[log_smah_sim[-1] - log_smah_sim < 0.1] = 0.5
    weight = jnp.array(weight)

    weight_fstar = np.ones_like(t_sim)
    weight_fstar[~mask] = 1e10
    weight_fstar = weight_fstar[fstar_indx_high]
    weight_fstar[log_fstar_sim.max() - log_fstar_sim < 0.1] = 0.5
    weight_fstar[weight_fstar == -10.0] = 1e10

    return weight, weight_fstar

