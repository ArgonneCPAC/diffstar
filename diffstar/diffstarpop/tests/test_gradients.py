"""Test that mc_diffstar_sfh_galpop has non-zero gradients w/r/t all its parameters"""

import numpy as np
from diffmah.diffmah_kernels import mah_halopop
from diffsky.mass_functions.mc_diffmah_tpeak import mc_subhalos
from diffstar.defaults import LGT0
from dsps.constants import T_TABLE_MIN
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from .. import get_bounded_diffstarpop_params, mc_diffstar_sfh_galpop
from ..defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DiffstarPopUParams,
)
from ..kernels.diffstarpop_mgash import _diffstarpop_means_covs


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


def get_random_dpp_params(ran_key, dp=0.1):
    collector = []
    ran_keys = jran.split(ran_key, len(DEFAULT_DIFFSTARPOP_U_PARAMS))
    for key, params in zip(ran_keys, DEFAULT_DIFFSTARPOP_U_PARAMS):
        u = jran.uniform(key, minval=-dp, maxval=dp, shape=(len(params),))
        ran_u_params = np.array(params) + u
        collector.append(params._make(ran_u_params))

    dpp_u_params = DiffstarPopUParams(*collector)
    dpp_params = get_bounded_diffstarpop_params(dpp_u_params)
    return dpp_params, dpp_u_params


def _check_grads_are_nonzero(grads):
    if_grads_close_to_zero = np.isclose(0.0, grads, rtol=1e-13, atol=1e-13)
    if if_grads_close_to_zero.any():
        raise AssertionError(
            "Parameters with exact zero gradients:",
            list(np.array(grads._fields)[if_grads_close_to_zero]),
        )


def _enforce_nonzero_grads(grads):
    assert np.all(np.isfinite(grads.u_sfh_pdf_cens_params))
    assert np.all(np.isfinite(grads.u_satquench_params))

    _check_grads_are_nonzero(grads.u_sfh_pdf_cens_params)
    _check_grads_are_nonzero(grads.u_satquench_params)


def test_all_diffstarpop_u_param_gradients_are_nonzero():
    """Verify that <SFH(t)> has nonzero gradient w/r/t all u_params"""

    ran_key = jran.PRNGKey(0)

    # Generate a random subhalo catalog
    subcat_key, ran_key = jran.split(ran_key, 2)
    lgmp_min = 11.25
    z_obs = 0.01
    Lbox = 75.0
    volume_com = Lbox**3
    subcat = mc_subhalos(subcat_key, z_obs, lgmp_min=lgmp_min, volume_com=volume_com)

    n_halos = subcat.logmhost_ult_inf.shape[0]

    lgmu_infall = subcat.logmp_ult_inf - subcat.logmhost_ult_inf
    gyr_since_infall = subcat.t_obs - subcat.t_ult_inf

    ntimes = 5
    tarr = np.linspace(T_TABLE_MIN, 13.7, ntimes)

    dmhdt_fit, log_mah_fit = mah_halopop(subcat.mah_params, tarr, LGT0)

    # compute SFHs for the default galaxy population
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        subcat.mah_params,
        subcat.logmp0,
        subcat.upids,
        lgmu_infall,
        subcat.logmhost_ult_inf,
        gyr_since_infall,
        ran_key,
        tarr,
    )

    (
        diffstar_params_ms,
        diffstar_params_q,
        default_sfh_ms,
        default_sfh_q,
        frac_q,
        mc_is_q,
    ) = mc_diffstar_sfh_galpop(*args)

    assert default_sfh_q.shape == (n_halos, ntimes)
    assert np.all(np.isfinite(default_sfh_q))

    # Set target <SFH(t)> according to the default galpop
    target_mean_sfh = np.mean(
        frac_q[:, None] * default_sfh_q + (1.0 - frac_q[:, None]) * default_sfh_ms,
        axis=0,
    )

    # Generate an alternate galpop at some other point in param space
    ran_params_key, ran_key = jran.split(ran_key, 2)
    alt_dpp_params, alt_dpp_u_params = get_random_dpp_params(ran_params_key)

    # Compute the SFH of the alternate galpop and verify it's well-behaved
    args = (
        alt_dpp_params,
        subcat.mah_params,
        subcat.logmp0,
        subcat.upids,
        lgmu_infall,
        subcat.logmhost_ult_inf,
        gyr_since_infall,
        ran_key,
        tarr,
    )
    # alt_diffstar_params, alt_sfh = mc_diffstar_sfh_galpop(*args)
    (
        alt_diffstar_params_ms,
        alt_diffstar_params_q,
        alt_sfh_ms,
        alt_sfh_q,
        alt_frac_q,
        mc_is_q,
    ) = mc_diffstar_sfh_galpop(*args)
    assert alt_sfh_q.shape == (n_halos, ntimes)
    assert np.all(np.isfinite(alt_sfh_q))

    # Define a dummy loss function based on the target <SFH(t)>
    @jjit
    def _loss(u_params):
        dpp = get_bounded_diffstarpop_params(u_params)
        args = (
            dpp,
            subcat.mah_params,
            subcat.logmp0,
            subcat.upids,
            lgmu_infall,
            subcat.logmhost_ult_inf,
            gyr_since_infall,
            ran_key,
            tarr,
        )
        # __, pred_sfh = mc_diffstar_sfh_galpop(*args)
        (
            pred_diffstar_params_ms,
            pred_diffstar_params_q,
            pred_sfh_ms,
            pred_sfh_q,
            pred_frac_q,
            mc_is_q,
        ) = mc_diffstar_sfh_galpop(*args)
        pred_mean_sfh_total = jnp.mean(
            pred_frac_q[:, None] * pred_sfh_q
            + (1.0 - pred_frac_q[:, None]) * pred_sfh_ms,
            axis=0,
        )

        return _mse(pred_mean_sfh_total, target_mean_sfh)

    loss_and_grad = value_and_grad(_loss)
    loss, loss_grads = loss_and_grad(alt_dpp_u_params)
    assert loss > 0
    _enforce_nonzero_grads(loss_grads)


def test_gradients_of_diffstarpop_pdf_satquench_params_are_nonzero():
    ran_key = jran.PRNGKey(0)

    logmp0 = 12.5
    tpeak = 12.0
    lgmu_infall = -1.5
    logmhost = 13.5
    gyr_since_infall = 1.0
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        logmp0,
        tpeak,
        lgmu_infall,
        logmhost,
        gyr_since_infall,
    )
    _res = _diffstarpop_means_covs(*args)
    frac_quench = _res[0]
    (
        frac_quench_sat,
        mu_mseq,
        mu_qseq,
        cov_mseq_ms_block,
        cov_qseq_ms_block,
        cov_qseq_q_block,
    ) = _res[1:]

    # Generate an alternate galpop at some other point in param space
    ran_params_key, ran_key = jran.split(ran_key, 2)
    alt_dpp_params, alt_dpp_u_params = get_random_dpp_params(ran_params_key, dp=1.0)

    args = (
        alt_dpp_params,
        logmp0,
        tpeak,
        lgmu_infall,
        logmhost,
        gyr_since_infall,
    )
    _res = _diffstarpop_means_covs(*args)
    frac_quench2 = _res[0]
    (
        frac_quench_sat2,
        mu_mseq2,
        mu_qseq2,
        cov_mseq_ms_block2,
        cov_qseq_ms_block2,
        cov_qseq_q_block2,
    ) = _res[1:]

    assert not np.allclose(mu_mseq2, mu_mseq)
    assert not np.allclose(cov_qseq_ms_block2, cov_qseq_ms_block)
    assert not np.allclose(cov_qseq_q_block2, cov_qseq_q_block)
    assert not np.allclose(frac_quench2, frac_quench)

    frac_q_target = np.copy(frac_quench)

    @jjit
    def _loss(u_params):
        dpp_params = get_bounded_diffstarpop_params(u_params)
        args = (
            dpp_params,
            logmp0,
            tpeak,
            lgmu_infall,
            logmhost,
            gyr_since_infall,
        )
        _res = _diffstarpop_means_covs(*args)
        frac_q_pred = _res[0]
        return _mse(frac_q_pred, frac_q_target)

    frac_q_loss, frac_q_grads = value_and_grad(_loss)(alt_dpp_u_params)
    assert np.isfinite(frac_q_loss)
    assert frac_q_loss > 1e-6
    _check_grads_are_nonzero(frac_q_grads.u_satquench_params)
