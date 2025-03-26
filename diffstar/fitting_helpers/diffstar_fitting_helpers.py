""" """

import numpy as np
from diffmah import mah_singlehalo
from diffmah.defaults import LGT0
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp

from .. import calc_sfh_singlegal
from ..defaults import (
    DEFAULT_DIFFSTAR_U_PARAMS,
    DEFAULT_MS_PARAMS,
    DEFAULT_Q_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    SFR_MIN,
    get_bounded_diffstar_params,
)
from ..kernels.main_sequence_kernels_tpeak import _get_unbounded_sfr_params
from ..kernels.quenching_kernels import _get_unbounded_q_params
from ..utils import _sigmoid, compute_fstar, cumulative_mstar_formed
from .utils import minimizer_wrapper

T_FIT_MIN = 1.0  # Only fit snapshots above this threshold. Gyr units.
DLOGM_CUT = 3.5  # Only fit SMH within this dex of the present day stellar mass.
MIN_MASS_CUT = 7.0  # Only fit SMH above this threshold. Log10(Msun) units.
FSTAR_TIME_DELAY = 1.0  # Time period of averaged SFH (aka fstar). Gyr units.
SSFRH_FLOOR = 1e-12  # Clip SFH to this minimum sSFR value. 1/yr units.


def diffstar_fitter(
    t_table,
    sfh_table,
    mah_params,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    mass_fit_min=MIN_MASS_CUT,
    fstar_tdelay=FSTAR_TIME_DELAY,
    ssfrh_floor=SSFRH_FLOOR,
    lgt0=LGT0,
    fb=FB,
):
    """Run the diffstar fitter on the input SFH"""
    u_p_init_and_err, loss_data = get_loss_data_default(
        t_table,
        sfh_table,
        mah_params,
        dlogm_cut=dlogm_cut,
        t_fit_min=t_fit_min,
        mass_fit_min=mass_fit_min,
        fstar_tdelay=fstar_tdelay,
        ssfrh_floor=ssfrh_floor,
        lgt0=lgt0,
        fb=fb,
    )
    _res = minimizer_wrapper(
        loss_default_clipssfrh,
        loss_grad_default_clipssfrh_np,
        u_p_init_and_err,
        loss_data,
    )
    varied_u_p_best, loss_best, success = _res

    # Transform varied_u_p_best into p_best
    u_indx_hi = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params.u_indx_hi
    u_p_best = (*varied_u_p_best[:3], u_indx_hi, *varied_u_p_best[3:])
    u_ms_params = DEFAULT_MS_PARAMS._make(u_p_best[:5])
    u_q_params = DEFAULT_Q_PARAMS._make(u_p_best[5:])
    u_p_best = DEFAULT_DIFFSTAR_U_PARAMS._make((u_ms_params, u_q_params))
    p_best = get_bounded_diffstar_params(u_p_best)

    return p_best, loss_best, success


def get_loss_data_default(
    t_table,
    sfh_table,
    mah_params,
    dlogm_cut=DLOGM_CUT,
    t_fit_min=T_FIT_MIN,
    mass_fit_min=MIN_MASS_CUT,
    fstar_tdelay=FSTAR_TIME_DELAY,
    ssfrh_floor=SSFRH_FLOOR,
    lgt0=LGT0,
    fb=FB,
):
    """Get loss data to use with diffstar_fitter"""
    sfh_target = np.clip(sfh_table, SFR_MIN, np.inf)
    mstar_target = cumulative_mstar_formed(t_table, sfh_table)
    logmstar_target = np.log10(mstar_target)

    fstar_table = compute_fstar(t_table, mstar_target, fstar_tdelay)
    ssfrh_table = fstar_table / mstar_target
    ssfrh_target = np.clip(ssfrh_table, ssfrh_floor, np.inf)

    fstar_target = ssfrh_target * mstar_target
    fstar_target_min = fstar_target.max() / 1000.0
    fstar_target = np.where(
        fstar_target < fstar_target_min, fstar_target_min, fstar_target
    )
    log_fstar_target = np.log10(fstar_target)

    lgt_table = jnp.log10(t_table)
    log_mah = mah_singlehalo(mah_params, t_table, lgt0)[1]
    logmp0 = log_mah[-1]

    weight, weight_fstar = get_weights(
        t_table,
        logmstar_target,
        log_fstar_target,
        fstar_tdelay,
        dlogm_cut,
        t_fit_min,
        mass_fit_min,
    )

    lgt_fstar_max = lgt_table[np.argmax(log_fstar_target)]

    ms_params = np.array(DEFAULT_MS_PARAMS)
    ms_params[0] = np.clip(0.3 * (logmp0 - 11.0) + 11.4, 11.0, 13.0)
    ms_params[1] = np.clip(0.2 * (logmp0 - 11.0) - 0.7, -1.5, -0.2)
    ms_params[2] = np.clip(0.7 * (logmp0 - 11.0) - 0.3, 0.2, 3.0)
    ms_params[4] = np.clip(-8.0 * (logmp0 - 11.0) + 15, 2.0, 15.0)
    ms_params = DEFAULT_MS_PARAMS._make(ms_params)
    u_ms_params = np.array(_get_unbounded_sfr_params(*ms_params))

    varied_u_ms_params = np.zeros(4)
    varied_u_ms_params[0:3] = u_ms_params[0:3]
    varied_u_ms_params[3] = u_ms_params[4]
    u_fixed_hi = u_ms_params[3]

    u_ms_params_err = np.array([0.5, 0.5, 1.0, 3.0])

    varied_q_params = np.array(DEFAULT_Q_PARAMS)
    varied_q_params[0] = np.clip(-0.5 * (logmp0 - 11.0) + 1.5, 0.7, 1.5)
    varied_q_params[2] = -2.0
    varied_q_params = DEFAULT_Q_PARAMS._make(DEFAULT_Q_PARAMS)
    varied_u_q_params = np.array(_get_unbounded_q_params(*varied_q_params))
    u_q_params_err = np.array([0.3, 0.5, 0.3, 0.3])

    loss_data = (
        t_table,
        mah_params,
        mstar_target,
        logmstar_target,
        sfh_target,
        log_fstar_target,
        fstar_tdelay,
        ssfrh_floor,
        weight,
        weight_fstar,
        lgt_fstar_max,
        u_fixed_hi,
        lgt0,
        fb,
    )

    u_p_init_and_err = (
        np.concatenate((varied_u_ms_params, varied_u_q_params)),
        np.concatenate((u_ms_params_err, u_q_params_err)),
    )
    return u_p_init_and_err, loss_data


def get_weights(
    t_table,
    log_smah_sim,
    log_fstar_sim,
    fstar_tdelay,
    dlogm_cut,
    t_fit_min,
    mass_fit_min,
):
    mass_fit_min = min(log_smah_sim[-1] - 0.5, mass_fit_min)

    mask = log_smah_sim > (log_smah_sim[-1] - dlogm_cut)
    mask &= log_smah_sim > mass_fit_min
    mask &= t_table >= t_fit_min

    weight = np.ones_like(t_table)
    weight[~mask] = 1e10
    weight[log_smah_sim[-1] - log_smah_sim < 0.1] = 0.5
    weight = jnp.array(weight)

    weight_fstar = np.ones_like(t_table)
    weight_fstar[~mask] = 1e10
    weight_fstar[log_fstar_sim.max() - log_fstar_sim < 0.1] = 0.5
    weight_fstar[weight_fstar == -10.0] = 1e10
    weight_fstar[t_table < fstar_tdelay + 0.01] = 1e10

    return weight, weight_fstar


@jjit
def loss_default_clipssfrh(u_params, loss_data):
    """
    MSE loss function for fitting individual stellar mass histories.
    The parameters k, indx_hi are fixed.

    """
    (
        t_table,
        mah_params,
        sm_target,
        log_sm_target,
        sfh_target,
        log_fstar_target,
        fstar_tdelay,
        ssfrh_floor,
        weight,
        weight_fstar,
        lgt_fstar_max,
        u_fixed_hi,
        lgt0,
        fb,
    ) = loss_data

    u_ms_params = [*u_params[0:3], u_fixed_hi, u_params[3]]
    u_ms_params = DEFAULT_U_MS_PARAMS._make(u_ms_params)
    u_q_params = u_params[4:8]
    u_q_params = DEFAULT_U_Q_PARAMS._make(u_q_params)

    sfh_u_params = DEFAULT_DIFFSTAR_U_PARAMS._make((u_ms_params, u_q_params))
    sfh_params = get_bounded_diffstar_params(sfh_u_params)
    sfh_table, mstar_table = calc_sfh_singlegal(
        sfh_params, mah_params, t_table, lgt0=lgt0, fb=fb, return_smh=True
    )

    fstar = compute_fstar(t_table, mstar_table, fstar_tdelay)
    fstar = jnp.clip(fstar, mstar_table * ssfrh_floor, jnp.inf)
    log_fstar = jnp.log10(fstar)

    logsm_table = jnp.log10(mstar_table)

    sfr_res = 1e8 * (sfh_table - sfh_target) / sm_target
    sfr_res = jnp.clip(sfr_res, -1.0, 1.0)

    loss = jnp.mean(((logsm_table - log_sm_target) / weight) ** 2)
    loss += jnp.mean(((log_fstar - log_fstar_target) / weight_fstar) ** 2)
    loss += jnp.mean((sfr_res / weight) ** 2)

    # Compute ridge terms
    loss += _sigmoid(sfh_params.q_params.lg_qt - lgt_fstar_max, 0.0, 50.0, 100.0, 0.0)
    loss += _sigmoid(sfh_params.ms_params.indx_lo, 0.0, 10.0, 1.0, 0.0)
    loss += _sigmoid(sfh_params.ms_params.lgy_at_mcrit, 0.0, 20.0, 0.0, 1.0)
    return loss


loss_grad_default_clipssfrh = jjit(grad(loss_default_clipssfrh, argnums=(0)))


def loss_grad_default_clipssfrh_np(params, data):
    return np.array(loss_grad_default_clipssfrh(params, data)).astype(float)
