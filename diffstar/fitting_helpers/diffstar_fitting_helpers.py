""" """

import warnings

import h5py
import numpy as np
from diffmah.defaults import LGT0
from diffmah.diffmah_kernels import _diffmah_kern
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp

from ..defaults import (
    DEFAULT_MS_PARAMS,
    DEFAULT_MS_PDICT,
    DEFAULT_Q_PARAMS,
    DEFAULT_Q_PDICT,
)
from ..kernels.main_sequence_kernels_tpeak import (
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
)
from ..kernels.quenching_kernels import (
    _get_bounded_q_params,
    _get_bounded_qt,
    _get_unbounded_q_params,
)
from ..utils import compute_fstar, cumulative_mstar_formed
from .fitting_kernels import calculate_sm_sfr_fstar_history_from_mah

T_FIT_MIN = 1.0  # Only fit snapshots above this threshold. Gyr units.
DLOGM_CUT = 3.5  # Only fit SMH within this dex of the present day stellar mass.
MIN_MASS_CUT = 7.0  # Only fit SMH above this threshold. Log10(Msun) units.
FSTAR_TIME_DELAY = 1.0  # Time period of averaged SFH (aka fstar). Gyr units.
SSFRH_FLOOR = 1e-12  # Clip SFH to this minimum sSFR value. 1/yr units.


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
):
    mstar_table = cumulative_mstar_formed(t_table, sfh_table)

    fstar_table = compute_fstar(t_table, mstar_table, fstar_tdelay)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssfrh = fstar_table / mstar_table
        ssfrh = np.clip(ssfrh, ssfrh_floor, np.inf)
        fstar_sim = ssfrh * mstar_table
        log_fstar_sim = np.where(
            fstar_sim == 0.0, np.log10(fstar_sim.max()) - 3.0, np.log10(fstar_sim)
        )

    logt = jnp.log10(t_table)
    dmhdt, log_mah = _diffmah_kern(mah_params, t_table, lgt0)

    weight, weight_fstar = get_weights(
        t_table,
        log_smah_sim,
        log_fstar_sim,
        fstar_tdelay,
        dlogm_cut,
        t_fit_min,
        mass_fit_min,
    )

    t_fstar_max = logt[np.argmax(log_fstar_sim)]

    default_sfr_params = np.array(DEFAULT_MS_PARAMS)
    default_sfr_params[0] = np.clip(0.3 * (logmp0 - 11.0) + 11.4, 11.0, 13.0)
    default_sfr_params[1] = np.clip(0.2 * (logmp0 - 11.0) - 0.7, -1.5, -0.2)
    default_sfr_params[2] = np.clip(0.7 * (logmp0 - 11.0) - 0.3, 0.2, 3.0)
    default_sfr_params[4] = np.clip(-8.0 * (logmp0 - 11.0) + 15, 2.0, 15.0)
    u_default_sfr_params = np.array(_get_unbounded_sfr_params(*default_sfr_params))

    sfr_ms_params = np.zeros(4)
    sfr_ms_params[0:3] = u_default_sfr_params[0:3]
    sfr_ms_params[3] = u_default_sfr_params[4]
    fixed_hi = u_default_sfr_params[3]

    sfr_ms_params_err = np.array([0.5, 0.5, 1.0, 3.0])

    default_q_params = np.array(DEFAULT_Q_PARAMS)
    default_q_params[0] = np.clip(-0.5 * (logmp0 - 11.0) + 1.5, 0.7, 1.5)
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
        sfh_table,
        log_fstar_sim,
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
