""" """

import numpy as np
from jax import random as jran

from ....defaults import DEFAULT_DIFFSTAR_PARAMS
from .. import diffstarpop_mgash as dsp
from ..defaults_mgash import DEFAULT_DIFFSTARPOP_PARAMS


def test_mc_diffstar_u_params_singlegal_kernel():
    logm0 = 13.0
    tpeak = 8.0
    upid = -1
    ran_key = jran.key(0)
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    _res = dsp.mc_diffstar_u_params_singlegal_kernel(*args)
    u_params_ms, u_params_qseq, frac_quench, mc_is_quenched_sequence = _res
    assert len(u_params_ms) == len(DEFAULT_DIFFSTAR_PARAMS)
    assert len(u_params_qseq) == len(DEFAULT_DIFFSTAR_PARAMS)
    for _u_p in u_params_ms:
        assert np.all(np.isfinite(_u_p))
    for _u_p in u_params_qseq:
        assert np.all(np.isfinite(_u_p))
    assert frac_quench.shape == ()
    assert mc_is_quenched_sequence.shape == ()


def test_diffstarpop_means_covs():
    logm0 = 13.0
    tpeak = 8.0
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    means_covs = dsp._diffstarpop_means_covs(
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    )
    #
    frac_quench = means_covs[0]
    assert np.all(frac_quench >= 0)
    assert np.all(frac_quench <= 1)
    mu_mseq = means_covs[1]
    assert np.all(np.isfinite(mu_mseq))
    cov_qseq_ms_block = means_covs[2]
    cov_qseq_q_block = means_covs[3]
    assert np.all(np.isfinite(cov_qseq_ms_block))
    assert np.all(np.isfinite(cov_qseq_q_block))


def test_mc_diffstar_u_params_singlegal_kernel_cen():
    logm0 = 13.0
    tpeak = 8.0
    ran_key = jran.key(0)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
        ran_key,
    )
    _res = dsp.mc_diffstar_u_params_singlegal_kernel_cen(*args)
    u_params_ms, u_params_qseq, frac_quench, mc_is_quenched_sequence = _res
    assert len(u_params_ms) == len(DEFAULT_DIFFSTAR_PARAMS)
    assert len(u_params_qseq) == len(DEFAULT_DIFFSTAR_PARAMS)
    for _u_p in u_params_ms:
        assert np.all(np.isfinite(_u_p))
    for _u_p in u_params_qseq:
        assert np.all(np.isfinite(_u_p))
    assert frac_quench.shape == ()
    assert mc_is_quenched_sequence.shape == ()


def test_diffstarpop_means_covs_cen():
    logm0 = 13.0
    tpeak = 8.0
    means_covs = dsp._diffstarpop_means_covs_cen(
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
    )
    #
    frac_quench = means_covs[0]
    assert np.all(frac_quench >= 0)
    assert np.all(frac_quench <= 1)
    mu_mseq = means_covs[1]
    assert np.all(np.isfinite(mu_mseq))
    cov_qseq_ms_block = means_covs[2]
    cov_qseq_q_block = means_covs[3]
    assert np.all(np.isfinite(cov_qseq_ms_block))
    assert np.all(np.isfinite(cov_qseq_q_block))


def get_randoms_from_key(ran_key):
    (
        ms_key_ms_block,
        q_key_ms_block,
        q_key_q_block,
        frac_q_key,
    ) = jran.split(ran_key, 4)

    random_normal_ms_ms_block = jran.normal(
        ms_key_ms_block,
        shape=(4,),
    )

    random_normal_q_ms_block = jran.normal(
        q_key_ms_block,
        shape=(4,),
    )

    random_normal_q_q_block = jran.normal(
        q_key_q_block,
        shape=(4,),
    )

    random_uniform_mc_is_q = jran.uniform(
        frac_q_key,
        minval=0,
        maxval=1,
        shape=(),
    )

    randoms = [
        random_normal_ms_ms_block,
        random_normal_q_ms_block,
        random_normal_q_q_block,
        random_uniform_mc_is_q,
    ]
    return randoms


def test_mc_diffstar_u_params_singlegal_from_randoms():
    logm0 = 13.0
    tpeak = 8.0
    upid = -1
    ran_key = jran.key(0)
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    args_default = [
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    ]
    args = tuple(args_default + [ran_key])
    res_key = dsp.mc_diffstar_u_params_singlegal_kernel(*args)

    randoms = get_randoms_from_key(ran_key)

    args = tuple(args_default + randoms)
    res_randoms = dsp.mc_diffstar_u_params_singlegal_kernel_from_randoms(*args)

    assert isinstance(res_key, dsp.MCDiffstarUParams)
    assert isinstance(res_randoms, dsp.MCDiffstarUParams)

    for x, y in zip(res_key, res_randoms):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


def test_mc_diffstar_u_params_singlegal_kernel_cen_from_randoms():
    logm0 = 13.0
    tpeak = 8.0
    ran_key = jran.key(0)
    args_default = [
        DEFAULT_DIFFSTARPOP_PARAMS,
        logm0,
        tpeak,
    ]
    args = tuple(args_default + [ran_key])
    res_key = dsp.mc_diffstar_u_params_singlegal_kernel_cen(*args)

    randoms = get_randoms_from_key(ran_key)

    args = tuple(args_default + randoms)
    res_randoms = dsp.mc_diffstar_u_params_singlegal_kernel_cen_from_randoms(*args)

    assert isinstance(res_key, dsp.MCDiffstarUParams)
    assert isinstance(res_randoms, dsp.MCDiffstarUParams)

    for x, y in zip(res_key, res_randoms):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
