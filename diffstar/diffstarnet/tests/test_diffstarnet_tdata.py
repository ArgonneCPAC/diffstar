"""
"""

import numpy as np
from jax import random as jran

from .. import diffstarnet_tdata as dtg


def enforce_good_tdata(tdata, logsm0_min=float("-inf")):
    for x in tdata:
        try:
            assert np.all(np.isfinite(x))
        except ValueError:  # x is a namedtuple
            for y in x:
                assert np.all(np.isfinite(y))

    logsm0 = np.log10(tdata.smh[:, -1])
    assert np.all(logsm0 >= logsm0_min)

    n_halos, n_times = tdata.sfh.shape
    n_halos2 = tdata.mah_params.logm0.size
    assert n_halos2 == n_halos

    history_keys = [x for x in dtg.SFH_KEYS if "params" not in x]

    for key in history_keys:
        arr = getattr(tdata, key)
        assert arr.shape == (n_halos, n_times)


def test_tdata_generator():
    ran_key = jran.key(0)
    n_halos = 5_000
    logm0_sample = np.linspace(10, 15, n_halos)

    # generate 5 epochs of data
    n_epochs = 5
    gen = dtg.tdata_generator(ran_key, logm0_sample, n_epochs=n_epochs)
    tdata_list = list(gen)
    assert len(tdata_list) == n_epochs
    for tdata in tdata_list:
        enforce_good_tdata(tdata, logsm0_min=dtg.LGSM0_MIN)

    # demo typical usage
    LOGSM0_MIN = 6.0
    gen = dtg.tdata_generator(ran_key, logm0_sample, n_epochs=2, logsm0_min=LOGSM0_MIN)
    tdata0 = next(gen)
    tdata1 = next(gen)
    try:
        next(gen)
    except StopIteration:
        pass  # expected because we tried to iterate for longer than n_epochs

    # tdata should not contain Mstar exceeding logsm0_min
    assert np.all(tdata0.smh[:, -1] >= 10**LOGSM0_MIN)
    assert np.all(tdata1.smh[:, -1] >= 10**LOGSM0_MIN)

    # tdata generator should yield different tdata with each iteration
    assert not np.allclose(tdata0.log_mah[0, :], tdata1.log_mah[0, :])
    assert not np.allclose(tdata0.sfh[0, :], tdata1.sfh[0, :])
    assert not np.allclose(tdata0.sfh_noq[0, :], tdata1.sfh_noq[0, :])
    assert not np.allclose(tdata0.sfh_noq_nolag[0, :], tdata1.sfh_noq_nolag[0, :])


def test_mc_diffmah_halo_sample():
    ran_key = jran.key(0)
    n_halos_init = 2_000
    logm0_sample = np.linspace(10, 15, n_halos_init)

    gen = dtg.tdata_generator(ran_key, logm0_sample)
    tdata = next(gen)

    n_halos = tdata.mah_params.logm0.size
    assert n_halos_init >= n_halos
    assert tdata.mah_params.t_peak.size == n_halos

    diff = tdata.mah_params.logm0 - tdata.log_mah[:, -1]
    assert np.abs(diff).mean() < 0.1
    assert np.std(diff) < 0.3
