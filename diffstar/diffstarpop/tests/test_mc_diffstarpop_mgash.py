""" """

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import mc_diffstarpop_mgash as mcdsp
from ..kernels.defaults_mgash import DEFAULT_DIFFSTARPOP_PARAMS


def test_mc_diffstar_params_singlegal_evaluates():
    logmp0 = 13.0
    tpeak = 8.0
    upid = -1
    ran_key = jran.PRNGKey(0)
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        logmp0,
        tpeak,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    _res = mcdsp.mc_diffstar_params_singlegal(*args)
    params_ms, params_qseq, frac_q, mc_is_q = _res
    assert np.all(frac_q >= 0)
    assert np.all(frac_q <= 1)
    assert np.all(np.isfinite(params_ms))
    assert np.all(np.isfinite(params_qseq))
    assert mc_is_q in (False, True)


def test_mc_diffstar_sfh_singlegal_evaluates():
    logmp0 = 13.0
    upid = -1
    ran_key = jran.PRNGKey(0)
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    n_times = 30
    tarr = np.linspace(0.1, 13.8, n_times)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        logmp0,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr,
    )
    _res = mcdsp.mc_diffstar_sfh_singlegal(*args)
    params_ms, params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res
    assert np.all(frac_q >= 0)
    assert np.all(frac_q <= 1)
    assert np.all(np.isfinite(params_ms))
    assert np.all(np.isfinite(params_q))
    assert mc_is_q in (False, True)
    assert sfh_q.shape == (n_times,)
    assert sfh_ms.shape == (n_times,)
    assert np.all(np.isfinite(sfh_q))
    assert np.all(np.isfinite(sfh_ms))
    assert np.all(sfh_ms > 0)
    assert np.all(sfh_q > 0)


def test_mc_diffstar_u_params_galpop():
    ngals = 50
    zz = np.zeros(ngals)
    logmp0 = 13.0 + zz
    tpeak_arr = np.random.uniform(1.0, 13.0, ngals)
    upids = np.random.choice([0, 1], ngals, replace=True)
    lgmu_infall = -1.0 + zz
    logmhost_infall = 13.0 + zz
    gyr_since_infall = 2.0 + zz
    ran_key = jran.key(0)
    _res = mcdsp.mc_diffstar_u_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        logmp0,
        tpeak_arr,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_u_params_ms, diffstar_u_params_q, frac_q, mc_is_q = _res
    assert np.all(np.isfinite(diffstar_u_params_ms))
    assert np.all(np.isfinite(diffstar_u_params_q))
    assert np.all(np.isfinite(frac_q))
    assert np.all(np.isfinite(mc_is_q))


def test_mc_diffstar_params_galpop():
    ngals = 50
    zz = np.zeros(ngals)
    logmp0 = np.zeros(ngals) + 12.0
    t_peak = zz + 10.0
    upid = zz - 1
    lgmu_infall = -1.0 + zz
    logmhost_infall = 13.0 + zz
    gyr_since_infall = 2.0 + zz
    ran_key = jran.key(0)
    _res = mcdsp.mc_diffstar_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        logmp0,
        t_peak,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_params_ms, diffstar_params_q, frac_q, mc_is_q = _res


def test_mc_diffstar_sfh_galpop():
    n_halos = 100
    ZZ = np.zeros(n_halos)

    ran_key = jran.PRNGKey(np.random.randint(2**32))
    lgmu_infall = -1.0 + ZZ
    logmhost_infall = 13.0 + ZZ
    gyr_since_infall = 2.0 + ZZ
    upids = np.random.choice([0, 1], n_halos, replace=True)

    t_table = np.linspace(1.0, 13.8, 100)

    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    logmp0 = np.random.uniform(low=11.0, high=15.0, size=(n_halos))
    mah_params = mah_params._replace(logm0=logmp0)
    mah_params = np.array(mah_params)

    _res = mcdsp.mc_diffstar_sfh_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    sfh_q, sfh_ms, frac_q = _res[2:5]

    assert np.isfinite(sfh_q).all()
    assert np.isfinite(sfh_ms).all()
    assert np.isfinite(frac_q).all()

    assert (sfh_q >= 0.0).all()
    assert (sfh_ms >= 0.0).all()
    assert (frac_q >= 0.0).all()
