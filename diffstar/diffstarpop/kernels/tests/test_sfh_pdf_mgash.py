""" """

import numpy as np
from jax import random as jran

from ...tests.test_utils import _enforce_is_cov
from .. import sfh_pdf_mgash as qseq

EPSILON = 1e-5


def test_param_u_param_names_propagate_properly():
    gen = zip(qseq.SFH_PDF_QUENCH_U_PARAMS._fields, qseq.SFH_PDF_QUENCH_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = qseq.get_bounded_sfh_pdf_params(
        qseq.SFH_PDF_QUENCH_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        qseq.SFH_PDF_QUENCH_PARAMS._fields
    )

    inferred_default_u_params = qseq.get_unbounded_sfh_pdf_params(
        qseq.SFH_PDF_QUENCH_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        qseq.SFH_PDF_QUENCH_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        qseq.get_bounded_sfh_pdf_params(qseq.SFH_PDF_QUENCH_PARAMS)
        raise NameError("get_bounded_sfh_pdf_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        qseq.get_unbounded_sfh_pdf_params(qseq.SFH_PDF_QUENCH_U_PARAMS)
        raise NameError("get_unbounded_sfh_pdf_params should not accept u_params")
    except AttributeError:
        pass


def test_get_qseq_means_and_covs_vmap_fails_when_passed_u_params():
    lgmarr = np.linspace(10, 15, 20)

    try:
        qseq._get_qseq_means_and_covs_vmap(
            qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS, lgmarr
        )
        raise NameError("_get_qseq_means_and_covs_vmap should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    ran_key = jran.key(0)
    n_tests = 100
    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        n_p = len(qseq.SFH_PDF_QUENCH_PARAMS)
        u_p = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_p,))
        u_p = qseq.QseqUParams(*u_p)
        p = qseq.get_bounded_sfh_pdf_params(u_p)
        u_p2 = qseq.get_unbounded_sfh_pdf_params(p)
        for x, y in zip(u_p, u_p2):
            assert np.allclose(x, y, rtol=0.01)


def test_covs_are_always_covs_default_params():
    for lgm in np.linspace(10, 15, 20):
        cov_qseq_q_block = qseq._get_covariance_qseq_q_block(
            qseq.SFH_PDF_QUENCH_PARAMS, lgm
        )
        assert cov_qseq_q_block.shape == (4, 4)
        _enforce_is_cov(cov_qseq_q_block)

        cov_qseq_ms_block = qseq._get_covariance_qseq_ms_block(
            qseq.SFH_PDF_QUENCH_PARAMS, lgm
        )
        assert cov_qseq_ms_block.shape == (4, 4)
        _enforce_is_cov(cov_qseq_ms_block)


def test_covs_are_always_covs_random_params():
    lgmarr = np.linspace(10, 15, 20)
    ran_key = jran.key(0)
    npars = len(qseq.SFH_PDF_QUENCH_PARAMS)
    ntests = 200
    for __ in range(ntests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_p = jran.uniform(test_key, minval=-5, maxval=5, shape=(npars,))
        u_params = qseq.QseqUParams(*u_p)
        params = qseq.get_bounded_sfh_pdf_params(u_params)

        for lgm in lgmarr:
            cov_qseq_ms_block = qseq._get_covariance_qseq_ms_block(params, lgm)
            assert cov_qseq_ms_block.shape == (4, 4)
            _enforce_is_cov(cov_qseq_ms_block)

            cov_qseq_q_block = qseq._get_covariance_qseq_q_block(params, lgm)
            assert cov_qseq_q_block.shape == (4, 4)
            _enforce_is_cov(cov_qseq_q_block)


def test_frac_quench_cen():
    lgmarr = np.linspace(1, 20, 100)
    tpeakarr = np.linspace(1.0, 20, 100)
    fqarr = qseq._frac_quench_cen(qseq.SFH_PDF_QUENCH_PARAMS, lgmarr, tpeakarr)
    assert np.all(fqarr >= 0.0)
    assert np.all(fqarr <= 1.0)


def test_frac_quench_sat():
    lgmarr = np.linspace(1, 20, 100)
    tpeakarr = np.linspace(1.0, 20, 100)
    fqarr = qseq._frac_quench_sat(qseq.SFH_PDF_QUENCH_PARAMS, lgmarr, tpeakarr)
    assert np.all(fqarr >= 0.0)
    assert np.all(fqarr <= 1.0)


def test_default_params_are_in_bounds():
    for key in qseq.SFH_PDF_QUENCH_PDICT.keys():
        bounds = qseq.SFH_PDF_QUENCH_BOUNDS_PDICT[key]
        val = qseq.SFH_PDF_QUENCH_PDICT[key]
        assert bounds[0] < val < bounds[1], key


def test_params_u_params_inverts():
    qseq_massonly_u_params = qseq.get_unbounded_sfh_pdf_params(
        qseq.SFH_PDF_QUENCH_PARAMS
    )
    qseq_massonly_params = qseq.get_bounded_sfh_pdf_params(qseq_massonly_u_params)
    assert np.allclose(qseq.SFH_PDF_QUENCH_PARAMS, qseq_massonly_params, rtol=5e-4)


def test_get_mean_u_params_mseq():
    lgmarr = np.linspace(11, 15, 100)
    params = qseq.SFH_PDF_QUENCH_PARAMS
    _means = qseq._get_mean_u_params_mseq(params, lgmarr)
    assert len(_means) == 4
    for x in _means:
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_mseq_block():
    n_gals = 50
    lgmarr = np.linspace(10, 15, n_gals)
    params = qseq.SFH_PDF_QUENCH_PARAMS
    _mean_pars_ms_block = qseq._get_mean_u_params_mseq(params, lgmarr)
    assert len(_mean_pars_ms_block) == 4
    for x in _mean_pars_ms_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_qseq():
    n_gals = 100
    lgmarr = np.linspace(11, 15, n_gals)
    tpeakarr = np.random.uniform(1.0, 20, n_gals)
    params = qseq.SFH_PDF_QUENCH_PARAMS
    _means = qseq._get_mean_u_params_qseq(params, lgmarr, tpeakarr)
    assert len(_means) == 8
    for x in _means:
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_qseq_block():
    n_gals = 50
    lgmarr = np.linspace(10, 15, n_gals)
    tpeakarr = np.random.uniform(1.0, 20, n_gals)

    params = qseq.SFH_PDF_QUENCH_PARAMS
    _mean_pars_ms_block = qseq._get_mean_u_params_qseq(params, lgmarr, tpeakarr)
    assert len(_mean_pars_ms_block) == 8
    for x in _mean_pars_ms_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))


def test_get_cov_params_mseq_ms_block():
    lgm = 13.0
    _res = qseq._get_cov_params_mseq_ms_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    diags, off_diags = _res
    ndim = 4
    assert len(diags) == ndim
    assert len(off_diags) == ndim * (ndim + 1) / 2 - ndim
    for x in diags:
        assert np.all(np.isfinite(x))
    for x in off_diags:
        assert np.all(np.isfinite(x))


def test_get_cov_params_qseq_ms_block():
    lgm = 13.0
    _res = qseq._get_cov_params_qseq_ms_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    diags, off_diags = _res
    ndim = 4
    assert len(diags) == ndim
    assert len(off_diags) == ndim * (ndim + 1) / 2 - ndim
    for x in diags:
        assert np.all(np.isfinite(x))
    for x in off_diags:
        assert np.all(np.isfinite(x))


def test_get_cov_params_qseq_q_block():
    lgm = 13.0
    _res = qseq._get_cov_params_qseq_q_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    diags, off_diags = _res
    ndim = 4
    assert len(diags) == ndim
    assert len(off_diags) == ndim * (ndim + 1) / 2 - ndim
    for x in diags:
        assert np.all(np.isfinite(x))
    for x in off_diags:
        assert np.all(np.isfinite(x))


def test_qseq_pdf_scalar_kernel():
    n_gals = 50
    lgmarr = np.linspace(10, 15, n_gals)
    tpeakarr = np.random.uniform(1.0, 20, n_gals)

    for i in range(n_gals):
        lgm = lgmarr[i]
        tpeak = tpeakarr[i]

        _res = qseq._sfh_pdf_scalar_kernel(qseq.SFH_PDF_QUENCH_PARAMS, lgm, tpeak)
        (
            frac_quench_cen,
            frac_quench_sat,
            mu_mseq,
            mu_qseq,
            cov_mseq_ms_block,
            cov_qseq_ms_block,
            cov_qseq_q_block,
        ) = _res
        for _x in _res:
            assert np.all(np.isfinite(_x))
        assert np.all(frac_quench_cen >= 0)
        assert np.all(frac_quench_cen <= 1)
        assert np.all(frac_quench_sat >= 0)
        assert np.all(frac_quench_sat <= 1)
        assert np.all(np.isfinite(mu_mseq))
        assert np.all(np.isfinite(mu_qseq))
        _enforce_is_cov(cov_mseq_ms_block)
        _enforce_is_cov(cov_qseq_ms_block)
        _enforce_is_cov(cov_qseq_q_block)
