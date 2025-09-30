"""
"""

import numpy as np
from jax import random as jran

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
    DEFAULT_SATQUENCHPOP_U_PARAMS,
    get_bounded_satquenchpop_params,
    get_qprob_sat,
    get_unbounded_satquenchpop_params,
    predict_sat_frac_delta_qprob,
)

TOL = 1e-2


def test_predict_sat_qprob_evaluates_on_defaults():
    n_tests = 100
    orig_key = jran.PRNGKey(0)
    ran_keys = jran.split(orig_key, n_tests)
    logmu_infall_lo, logmu_infall_hi = -5, 1.0
    logmhost_infall_lo, logmhost_infall_hi = 8.0, 16.0
    gyr_since_infall_lo, gyr_since_infall_hi = -5.0, 25.0

    for ran_key in ran_keys:
        res = jran.uniform(ran_key, shape=(3,))
        logmu_infall = logmu_infall_lo + res[0] * (logmu_infall_hi - logmu_infall_lo)
        logmhost_infall = logmhost_infall_lo + res[1] * (
            logmhost_infall_hi - logmhost_infall_lo
        )
        gyr_since_infall = gyr_since_infall_lo + res[2] * (
            gyr_since_infall_hi - gyr_since_infall_lo
        )

        frac_delta_qprob = predict_sat_frac_delta_qprob(
            DEFAULT_SATQUENCHPOP_PARAMS, logmu_infall, logmhost_infall, gyr_since_infall
        )
        assert 0 <= frac_delta_qprob <= 1


def test_get_qprob_sat_evaluates_on_defaults():
    n_tests = 100
    orig_key = jran.PRNGKey(0)
    ran_keys = jran.split(orig_key, n_tests)
    logmu_infall_lo, logmu_infall_hi = -5, 1.0
    logmhost_infall_lo, logmhost_infall_hi = 8.0, 16.0
    gyr_since_infall_lo, gyr_since_infall_hi = -5.0, 25.0

    for ran_key in ran_keys:
        res = jran.uniform(ran_key, shape=(3,))
        logmu_infall = logmu_infall_lo + res[0] * (logmu_infall_hi - logmu_infall_lo)
        logmhost_infall = logmhost_infall_lo + res[1] * (
            logmhost_infall_hi - logmhost_infall_lo
        )
        gyr_since_infall = gyr_since_infall_lo + res[2] * (
            gyr_since_infall_hi - gyr_since_infall_lo
        )
        qprob_cen = res[3]

        qprob_sat = get_qprob_sat(
            DEFAULT_SATQUENCHPOP_PARAMS,
            logmu_infall,
            logmhost_infall,
            gyr_since_infall,
            qprob_cen,
        )
        assert 0 <= qprob_sat <= 1


def test_param_u_param_names_propagate_properly():
    gen = zip(
        DEFAULT_SATQUENCHPOP_U_PARAMS._fields, DEFAULT_SATQUENCHPOP_PARAMS._fields
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_satquenchpop_params(
        DEFAULT_SATQUENCHPOP_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        DEFAULT_SATQUENCHPOP_PARAMS._fields
    )

    inferred_default_u_params = get_unbounded_satquenchpop_params(
        DEFAULT_SATQUENCHPOP_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_SATQUENCHPOP_U_PARAMS._fields
    )


def test_get_bounded_satquenchpop_params_fails_when_passing_params():
    try:
        get_bounded_satquenchpop_params(DEFAULT_SATQUENCHPOP_PARAMS)
        raise NameError("get_bounded_satquenchpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_satquenchpop_params_fails_when_passing_u_params():
    try:
        get_unbounded_satquenchpop_params(DEFAULT_SATQUENCHPOP_U_PARAMS)
        raise NameError("get_bounded_satquenchpop_u_params should not accept u_params")
    except AttributeError:
        pass


def test_get_qprob_sat_fails_when_passing_u_params():
    lgmu_infall, logmhost_infall, gyr_since_infall, qprob_cen = -2.0, 13.0, 0.0, 0.5
    data = lgmu_infall, logmhost_infall, gyr_since_infall, qprob_cen

    try:
        get_qprob_sat(DEFAULT_SATQUENCHPOP_U_PARAMS, *data)
        raise NameError("get_qprob_sat should not accept u_params")
    except AttributeError:
        pass


def test_satquenchpop_u_param_inversion_default_params():
    assert np.allclose(
        DEFAULT_SATQUENCHPOP_PARAMS,
        get_bounded_satquenchpop_params(DEFAULT_SATQUENCHPOP_U_PARAMS),
        rtol=TOL,
    )
    assert np.allclose(
        DEFAULT_SATQUENCHPOP_U_PARAMS,
        get_unbounded_satquenchpop_params(DEFAULT_SATQUENCHPOP_PARAMS),
        rtol=TOL,
    )
