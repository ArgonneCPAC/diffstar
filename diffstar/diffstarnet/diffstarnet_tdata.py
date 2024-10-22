"""
"""

from collections import namedtuple

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import _mc_diffmah_singlecen_vmap_kern
from jax import numpy as jnp
from jax import random as jran

from ..defaults import (
    DEFAULT_DIFFSTAR_U_PARAMS,
    DEFAULT_Q_PARAMS_UNQUENCHED,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    LGT0,
    T_TABLE_MIN,
    get_bounded_diffstar_params,
)
from ..kernels.main_sequence_kernels_tpeak import MS_PARAM_BOUNDS_PDICT
from ..sfh_model_tpeak import calc_sfh_galpop

T0 = 10**LGT0
N_SFH_TABLE = 200
LGMH_MIN = 10.5
LGSM0_MIN = 5.0

TAU_INST = MS_PARAM_BOUNDS_PDICT["tau_dep"][0] + 1e-4

_TDATA_SFH_ROOTKEYS = ["sfh_params", "sfh", "smh"]
_TDATA_NOQ_KEYS = [key + "_noq" for key in _TDATA_SFH_ROOTKEYS]
_TDATA_NOQ_NOLAG_KEYS = [key + "_nolag" for key in _TDATA_NOQ_KEYS]
SFH_KEYS = _TDATA_SFH_ROOTKEYS + _TDATA_NOQ_KEYS + _TDATA_NOQ_NOLAG_KEYS
TDATA_KEYS = ["mah_params", "log_mah"] + SFH_KEYS
TData = namedtuple("TData", TDATA_KEYS)


def tdata_generator(
    ran_key,
    logm0_sample,
    n_sfh_table=N_SFH_TABLE,
    logsm0_min=LGSM0_MIN,
    n_epochs=float("inf"),
):
    """Training data generator for diffstarnet

    Parameters
    ----------
    ran_key : jax.random.key

    logm0_sample : array, shape (n_halos, )
        Array of values of diffmah parameter logm0

    logsm0_min : float, optional
        Minimum z=0 stellar mass in the training data
        Default is set by LGSM0_MIN at top of module

    n_epochs : int, optional
        maximum number of batches to yield. Default is infinite

    Yields
    ------
    tdata : namedtuple
        mah_params, t_peak, log_mah, sfh_params, sfh, smh
        sfh_params_noq, sfh_noq, smh_noq
        sfh_params_noq_nolag, sfh_noq_nolag, smh_noq_nolag

    """
    batchnum = 0
    while batchnum < n_epochs:
        ran_key, batch_key = jran.split(ran_key, 2)
        tdata = _compute_tdata(batch_key, logm0_sample, n_sfh_table, logsm0_min)
        yield tdata
        batchnum += 1


def _compute_tdata(
    ran_key, logm0_sample, n_sfh_table=N_SFH_TABLE, logsm0_min=LGSM0_MIN
):
    """"""
    tarr = np.linspace(T_TABLE_MIN, T0, n_sfh_table)

    mah_key, early_late_key, sfh_key = jran.split(ran_key, 3)

    _reslist = mc_diffmah_halo_sample(mah_key, tarr, logm0_sample)
    mah_params_early, dmhdt_early, log_mah_early = _reslist[:3]
    mah_params_late, dmhdt_late, log_mah_late = _reslist[3:6]
    frac_early = _reslist[6]

    n_halos = mah_params_early.logm0.size
    uran_mah = jran.uniform(early_late_key, minval=0, maxval=1, shape=(n_halos,))
    msk_mah = frac_early < uran_mah
    mah_params = DEFAULT_MAH_PARAMS._make(
        [
            jnp.where(
                msk_mah, getattr(mah_params_early, key), getattr(mah_params_late, key)
            )
            for key in mah_params_late._fields
        ]
    )

    log_mah = jnp.where(msk_mah.reshape((-1, 1)), log_mah_early, log_mah_late)

    ZZ = np.zeros(n_halos)

    uran = jran.uniform(sfh_key, minval=-100, maxval=100, shape=(8, n_halos))

    u_ms_late_index = np.zeros(n_halos) + DEFAULT_U_MS_PARAMS.u_indx_hi
    u_ms_params = [uran[0], uran[1], uran[2], u_ms_late_index, uran[3]]
    u_ms_params = DEFAULT_U_MS_PARAMS._make(u_ms_params)
    u_q_params = DEFAULT_U_Q_PARAMS._make([uran[i, :] for i in range(4, 8)])

    sfh_u_params = DEFAULT_DIFFSTAR_U_PARAMS._make((u_ms_params, u_q_params))
    sfh_params = get_bounded_diffstar_params(sfh_u_params)

    q_params_noq = u_q_params._make([ZZ + x for x in DEFAULT_Q_PARAMS_UNQUENCHED])
    sfh_params_noq = sfh_params._replace(q_params=q_params_noq)

    ms_params_nolag = sfh_params.ms_params._replace(tau_dep=TAU_INST + ZZ)
    sfh_params_noq_nolag = sfh_params_noq._replace(ms_params=ms_params_nolag)

    sfh, smh = calc_sfh_galpop(
        sfh_params,
        mah_params,
        tarr,
        lgt0=LGT0,
        fb=FB,
        return_smh=True,
    )
    logsm0 = np.log10(smh)[:, -1]

    sfh_noq, smh_noq = calc_sfh_galpop(
        sfh_params_noq,
        mah_params,
        tarr,
        lgt0=LGT0,
        fb=FB,
        return_smh=True,
    )
    logsm0_noq = np.log10(smh_noq)[:, -1]

    sfh_noq_nolag, smh_noq_nolag = calc_sfh_galpop(
        sfh_params_noq_nolag,
        mah_params,
        tarr,
        lgt0=LGT0,
        fb=FB,
        return_smh=True,
    )
    logsm0_noq_nolag = np.log10(smh_noq_nolag)[:, -1]

    # Implement stellar mass cut
    msk = (
        (logsm0 > logsm0_min)
        & (logsm0_noq > logsm0_min)
        & (logsm0_noq_nolag > logsm0_min)
    )

    mah_params_out = mah_params._make([x[msk] for x in mah_params])
    log_mah_out = log_mah[msk]
    sfh_out = sfh[msk]
    smh_out = smh[msk]
    sfh_noq_out = sfh_noq[msk]
    smh_noq_out = smh_noq[msk]
    sfh_noq_nolag_out = sfh_noq_nolag[msk]
    smh_noq_nolag_out = smh_noq_nolag[msk]

    ms_params = sfh_params.ms_params._make([x[msk] for x in sfh_params.ms_params])
    q_params = sfh_params.q_params._make([x[msk] for x in sfh_params.q_params])
    sfh_params_out = sfh_params._make((ms_params, q_params))

    ms_params_noq_out = sfh_params_noq.ms_params._make(
        [x[msk] for x in sfh_params_noq.ms_params]
    )
    q_params_noq_out = sfh_params_noq.q_params._make(
        [x[msk] for x in sfh_params_noq.q_params]
    )
    sfh_params_noq_out = sfh_params_noq._make((ms_params_noq_out, q_params_noq_out))

    ms_params_noq_nolag_out = sfh_params_noq_nolag.ms_params._make(
        [x[msk] for x in sfh_params_noq_nolag.ms_params]
    )
    sfh_params_noq_nolag_out = sfh_params_noq_out._replace(
        ms_params=ms_params_noq_nolag_out
    )

    return TData(
        mah_params_out,
        log_mah_out,
        sfh_params_out,
        sfh_out,
        smh_out,
        sfh_params_noq_out,
        sfh_noq_out,
        smh_noq_out,
        sfh_params_noq_nolag_out,
        sfh_noq_nolag_out,
        smh_noq_nolag_out,
    )


def mc_diffmah_halo_sample(ran_key, tarr, logm0_sample):
    n_halos = logm0_sample.size
    ZZ = np.zeros(n_halos)
    t_0 = tarr[-1]
    t_obs = t_0 + ZZ
    lgt0 = np.log10(t_0)
    ran_keys = jran.split(ran_key, n_halos)
    _reslist = _mc_diffmah_singlecen_vmap_kern(
        DEFAULT_DIFFMAHPOP_PARAMS, tarr, logm0_sample, t_obs, ran_keys, lgt0
    )
    mah_params_early, dmhdt_early, log_mah_early = _reslist[:3]
    mah_params_late, dmhdt_late, log_mah_late = _reslist[3:6]
    frac_early = _reslist[6]
    return (
        mah_params_early,
        dmhdt_early,
        log_mah_early,
        mah_params_late,
        dmhdt_late,
        log_mah_late,
        frac_early,
    )
