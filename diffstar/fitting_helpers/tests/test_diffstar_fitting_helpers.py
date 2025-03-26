""" """

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...defaults import T_TABLE_MIN
from ...utils import cumulative_mstar_formed
from .. import diffstar_fitting_helpers as dfh


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
            u_fixed_hi,
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
        assert np.all(log_fstar_table > -20)
        assert np.all(log_fstar_table < 0)

        assert 10**lgt_fstar_max > t_table[0]
        assert 10**lgt_fstar_max < t_table[-1]

        assert np.allclose(10**lgt0, t0_sim)
        assert np.allclose(fb, fb_sim)
