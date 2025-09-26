""" """

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...defaults import (
    DEFAULT_DIFFSTAR_U_PARAMS,
    T_TABLE_MIN,
    get_bounded_diffstar_params,
)
from ...sfh_model import calc_sfh_singlegal
from ...utils import cumulative_mstar_formed
from .. import diffstar_fitting_helpers as dfh

LOSS_TOL = 0.1


def _get_random_diffstar_params(ran_key):
    ms_key, q_key = jran.split(ran_key, 2)
    u_ms = jran.uniform(
        ms_key,
        minval=-1,
        maxval=1,
        shape=(len(DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params),),
    )
    u_q = jran.uniform(
        ms_key,
        minval=-1,
        maxval=1,
        shape=(len(DEFAULT_DIFFSTAR_U_PARAMS.u_q_params),),
    )

    u_ms_params = u_ms + np.array(DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params)
    u_q_params = u_q + np.array(DEFAULT_DIFFSTAR_U_PARAMS.u_q_params)

    u_ms_params = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params._make(u_ms_params)
    u_q_params = DEFAULT_DIFFSTAR_U_PARAMS.u_q_params._make(u_q_params)

    u_sfh_params = DEFAULT_DIFFSTAR_U_PARAMS._make((u_ms_params, u_q_params))

    return u_sfh_params


def _mae(pred, target):
    diff = pred - target
    abserr = np.abs(diff)
    return np.mean(abserr)


def test_diffstar_fitter():
    ran_key = jran.key(0)
    n_tests = 100
    n_times = 50
    t0_sim = 13.6
    fb_sim = 0.15
    t_table = np.linspace(T_TABLE_MIN, t0_sim, n_times)

    loss_collector = []
    for __ in range(n_tests):

        ran_key, u_p_key = jran.split(ran_key, 2)
        u_sfh_params = _get_random_diffstar_params(u_p_key)
        sfh_params = get_bounded_diffstar_params(u_sfh_params)

        sfh_table, mstar_table = calc_sfh_singlegal(
            sfh_params,
            DEFAULT_MAH_PARAMS,
            t_table,
            lgt0=np.log10(t0_sim),
            fb=fb_sim,
            return_smh=True,
        )

        p_best, loss_best, success = dfh.diffstar_fitter(
            t_table, sfh_table, DEFAULT_MAH_PARAMS, lgt0=np.log10(t0_sim), fb=fb_sim
        )
        assert success == 1

        sfh_table_best, mstar_table_best = calc_sfh_singlegal(
            p_best,
            DEFAULT_MAH_PARAMS,
            t_table,
            lgt0=np.log10(t0_sim),
            fb=fb_sim,
            return_smh=True,
        )

        msk_t_fit = t_table > dfh.T_FIT_MIN
        logsm_table = np.log10(mstar_table)[msk_t_fit]
        logsm_table_best = np.log10(mstar_table_best)[msk_t_fit]
        mean_abs_err = _mae(logsm_table, logsm_table_best)
        loss_collector.append(mean_abs_err)

    assert np.mean(np.array(loss_collector) < LOSS_TOL) > 0.9


def test_loss_default_clipssfrh():
    ran_key = jran.key(0)
    n_tests = 100
    n_times = 200
    t0_sim = 13.6
    fb_sim = 0.15
    t_table = np.linspace(T_TABLE_MIN, t0_sim, n_times)

    for __ in range(n_tests):
        ran_key, sfh_key = jran.split(ran_key, 2)
        sfh_table = jran.uniform(sfh_key, minval=0, maxval=100, shape=(n_times,))

        _res = dfh.get_loss_data_default(
            t_table, sfh_table, DEFAULT_MAH_PARAMS, lgt0=np.log10(t0_sim), fb=fb_sim
        )
        u_p_init_and_err, loss_data = _res
        u_p_init, u_p_init_err = u_p_init_and_err
        loss_init = dfh.loss_default_clipssfrh(u_p_init, loss_data)
        assert np.all(np.isfinite(loss_init))
        assert loss_init > 0
        assert loss_init < 1_000.0

        loss_grads = dfh.loss_grad_default_clipssfrh(u_p_init, loss_data)
        assert np.all(np.isfinite(loss_grads))
        assert not np.any(np.isclose(loss_grads, 0.0))


def test_get_loss_data_default():
    ran_key = jran.key(0)
    n_tests = 100
    n_times = 200
    t0_sim = 13.6
    fb_sim = 0.15
    t_table = np.linspace(T_TABLE_MIN, t0_sim, n_times)

    for __ in range(n_tests):
        ran_key, sfh_key = jran.split(ran_key, 2)
        sfh_table = jran.uniform(sfh_key, minval=0, maxval=100, shape=(n_times,))

        _res = dfh.get_loss_data_default(
            t_table, sfh_table, DEFAULT_MAH_PARAMS, lgt0=np.log10(t0_sim), fb=fb_sim
        )
        u_p_init_and_err, loss_data = _res
        u_p_init, u_p_init_err = u_p_init_and_err
        assert np.all(np.isfinite(u_p_init))
        assert np.all(np.isfinite(u_p_init_err))
        assert u_p_init.shape == u_p_init_err.shape

        (
            t_table,
            mah_params,
            mstar_table,
            logmstar_table,
            sfh_table,
            log_fstar_table,
            fstar_tdelay,
            ssfrh_floor,
            weight,
            weight_fstar,
            lgt_fstar_max,
            lgt0,
            fb,
        ) = loss_data

        assert np.allclose(mstar_table, 10**logmstar_table)
        mstar_table2 = cumulative_mstar_formed(t_table, sfh_table)
        assert np.allclose(mstar_table2, mstar_table)
        assert weight.shape == (n_times,)
        assert np.all(np.isfinite(weight))
        assert np.all(weight >= 0)

        assert weight_fstar.shape == (n_times,)
        assert np.all(np.isfinite(weight_fstar))
        assert np.all(weight_fstar >= 0)

        assert np.all(np.isfinite(log_fstar_table))
        assert np.all(log_fstar_table > -10)
        assert np.all(log_fstar_table < np.log10(sfh_table.max()))

        assert 10**lgt_fstar_max > t_table[0]
        assert 10**lgt_fstar_max < t_table[-1]

        assert np.allclose(10**lgt0, t0_sim)
        assert np.allclose(fb, fb_sim)
