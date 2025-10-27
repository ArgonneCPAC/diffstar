""" """

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from diffstar.defaults import (
    DEFAULT_Q_U_PARAMS_UNQUENCHED,
    DiffstarUParams,
    MSUParams,
    QUParams,
)

from .defaults_mgash_ecrit_qt import DEFAULT_SATQUENCHPOP_PARAMS, SFH_PDF_QUENCH_PARAMS
from .satquenchpop_model import get_qprob_sat
from .sfh_pdf_mgash_ecrit_qt import _sfh_pdf_scalar_kernel


@jjit
def mc_diffstar_u_params_singlegal_kernel(
    diffstarpop_params,
    logmp0,
    tpeak,
    upid,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    means_covs = _diffstarpop_means_covs(
        diffstarpop_params,
        logmp0,
        tpeak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    )

    (
        frac_quench_cen,
        frac_quench_sat,
        mu_mseq,
        mu_qseq,
        cov_mseq_ms_block,
        cov_qseq_ms_block,
        cov_qseq_q_block,
    ) = means_covs

    frac_quench = jnp.where(upid == -1, frac_quench_cen, frac_quench_sat)
    mu_qseq_ms_block = mu_qseq[:4]
    mu_qseq_q_block = mu_qseq[4:]

    ms_key_ms_block, q_key_ms_block, q_key_q_block, frac_q_key = jran.split(ran_key, 4)

    u_params_mseq_ms_block = jran.multivariate_normal(
        ms_key_ms_block, jnp.array(mu_mseq), cov_mseq_ms_block, shape=()
    )
    u_params_qseq_ms_block = jran.multivariate_normal(
        q_key_ms_block, jnp.array(mu_qseq_ms_block), cov_qseq_ms_block, shape=()
    )
    u_params_qseq_q_block = jran.multivariate_normal(
        q_key_q_block, jnp.array(mu_qseq_q_block), cov_qseq_q_block, shape=()
    )
    u_params_q = jnp.array(
        (
            *u_params_qseq_ms_block[:4],
            *u_params_qseq_q_block,
        )
    )
    u_params_q = DiffstarUParams(
        *MSUParams(*u_params_q[:4]), *QUParams(*u_params_q[4:])
    )

    u_params_ms = jnp.array(
        (
            *u_params_mseq_ms_block[:4],
            *DEFAULT_Q_U_PARAMS_UNQUENCHED,
        )
    )
    u_params_ms = DiffstarUParams(
        *MSUParams(*u_params_ms[:4]), *QUParams(*u_params_ms[4:])
    )

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    mc_is_quenched_sequence = uran < frac_quench

    return u_params_ms, u_params_q, frac_quench, mc_is_quenched_sequence


@jjit
def _diffstarpop_means_covs(
    diffstarpop_params, logmp0, tpeak, lgmu_infall, logmhost_infall, gyr_since_infall
):
    sfh_pdf_cens_params = SFH_PDF_QUENCH_PARAMS._make(
        [getattr(diffstarpop_params, x) for x in SFH_PDF_QUENCH_PARAMS._fields]
    )
    means_covs = _sfh_pdf_scalar_kernel(sfh_pdf_cens_params, logmp0, tpeak)

    # Modify frac_q for satellites
    frac_q = means_covs[0]
    satquench_params = DEFAULT_SATQUENCHPOP_PARAMS._make(
        [getattr(diffstarpop_params, x) for x in DEFAULT_SATQUENCHPOP_PARAMS._fields]
    )
    frac_q = get_qprob_sat(
        satquench_params, lgmu_infall, logmhost_infall, gyr_since_infall, frac_q
    )
    means_covs = (frac_q, *means_covs[1:])
    return means_covs


@jjit
def _diffstarpop_means_covs_cen(diffstarpop_params, logmp0, tpeak):
    sfh_pdf_cens_params = SFH_PDF_QUENCH_PARAMS._make(
        [getattr(diffstarpop_params, x) for x in SFH_PDF_QUENCH_PARAMS._fields]
    )
    means_covs = _sfh_pdf_scalar_kernel(sfh_pdf_cens_params, logmp0, tpeak)

    return means_covs


@jjit
def mc_diffstar_u_params_singlegal_kernel_cen(
    diffstarpop_params, logmp0, tpeak, ran_key
):
    means_covs = _diffstarpop_means_covs_cen(diffstarpop_params, logmp0, tpeak)

    (
        frac_quench_cen,
        frac_quench_sat,
        mu_mseq,
        mu_qseq,
        cov_mseq_ms_block,
        cov_qseq_ms_block,
        cov_qseq_q_block,
    ) = means_covs

    frac_quench = frac_quench_cen
    mu_qseq_ms_block = mu_qseq[:4]
    mu_qseq_q_block = mu_qseq[4:]

    ms_key_ms_block, q_key_ms_block, q_key_q_block, frac_q_key = jran.split(ran_key, 4)

    u_params_mseq_ms_block = jran.multivariate_normal(
        ms_key_ms_block, jnp.array(mu_mseq), cov_mseq_ms_block, shape=()
    )
    u_params_qseq_ms_block = jran.multivariate_normal(
        q_key_ms_block, jnp.array(mu_qseq_ms_block), cov_qseq_ms_block, shape=()
    )
    u_params_qseq_q_block = jran.multivariate_normal(
        q_key_q_block, jnp.array(mu_qseq_q_block), cov_qseq_q_block, shape=()
    )
    u_params_q = jnp.array(
        (
            *u_params_qseq_ms_block[:4],
            *u_params_qseq_q_block,
        )
    )
    u_params_q = DiffstarUParams(
        *MSUParams(*u_params_q[:4]), *QUParams(*u_params_q[4:])
    )

    u_params_ms = jnp.array(
        (
            *u_params_mseq_ms_block[:4],
            *DEFAULT_Q_U_PARAMS_UNQUENCHED,
        )
    )
    u_params_ms = DiffstarUParams(
        *MSUParams(*u_params_ms[:4]), *QUParams(*u_params_ms[4:])
    )

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    mc_is_quenched_sequence = uran < frac_quench

    return u_params_ms, u_params_q, frac_quench, mc_is_quenched_sequence
