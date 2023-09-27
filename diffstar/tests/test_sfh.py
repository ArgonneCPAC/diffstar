"""
"""
import numpy as np

from .. import sfh_galpop, sfh_singlegal
from ..defaults import (
    DEFAULT_MS_PARAMS,
    DEFAULT_Q_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_U_Q_PARAMS,
    FB,
    LGT0,
    SFR_MIN,
)
from ..kernels.kernel_builders import get_sfh_from_mah_kern
from ..kernels.main_sequence_kernels import _get_unbounded_sfr_params_vmap
from ..kernels.quenching_kernels import _get_unbounded_q_params_vmap
from .test_gas import _get_default_mah_params


def _get_all_default_params():
    ms_params, q_params = DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return lgt0, mah_params, ms_params, q_params


def _get_all_default_u_params():
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return lgt0, mah_params, u_ms_params, u_q_params


def test_sfh_singlegal_evaluates():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
    assert sfh.shape == (n_t,)


def test_sfh_singlegal_agrees_with_kernel_builder():
    kern = get_sfh_from_mah_kern(tobs_loop="scan")

    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    __, __, u_ms_params, u_q_params = _get_all_default_u_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB)
    sfh2 = kern(tarr, mah_params, u_ms_params, u_q_params, LGT0, FB)

    assert np.allclose(sfh, sfh2, rtol=1e-4)


def test_sfh_singlegal_evaluates_with_unbounded_option():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    __, __, u_ms_params, u_q_params = _get_all_default_u_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB)
    u_sfh = sfh_singlegal(
        tarr,
        mah_params,
        u_ms_params,
        u_q_params,
        LGT0,
        FB,
        ms_param_type="unbounded",
        q_param_type="unbounded",
    )

    assert np.allclose(sfh, u_sfh, rtol=1e-3)


def test_sfh_galpop_evaluates():
    n_t = 100
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    mah_params = np.array(mah_params).reshape((1, -1))
    ms_params = np.array(ms_params).reshape((1, -1))
    q_params = np.array(q_params).reshape((1, -1))
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, LGT0, FB)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
    assert sfh.shape == (1, n_t)


def test_sfh_galpop_agrees_with_kernel_builder():
    kern = get_sfh_from_mah_kern(tobs_loop="scan", galpop_loop="vmap")

    n_t = 100
    lgt0, mah_params, u_ms_params, u_q_params = _get_all_default_params()
    mah_params = np.array(mah_params).reshape((1, -1))
    u_ms_params = np.array(u_ms_params).reshape((1, -1))
    u_q_params = np.array(u_q_params).reshape((1, -1))
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_galpop(
        tarr,
        mah_params,
        u_ms_params,
        u_q_params,
        LGT0,
        FB,
        ms_param_type="unbounded",
        q_param_type="unbounded",
    )
    sfh2 = kern(tarr, mah_params, u_ms_params, u_q_params, LGT0, FB)

    assert np.allclose(sfh, sfh2, rtol=1e-4)


def test_sfh_galpop_evaluates_with_bounded_option():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()
    ms_params, q_params = DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS

    mah_params = np.array(mah_params).reshape((1, -1))
    ms_params = np.array(ms_params).reshape((1, -1))
    q_params = np.array(q_params).reshape((1, -1))
    ms_params = np.array(ms_params).reshape((1, -1))
    q_params = np.array(q_params).reshape((1, -1))

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    u_sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, LGT0, FB)

    u_ms_params = _get_unbounded_sfr_params_vmap(ms_params)
    u_q_params = _get_unbounded_q_params_vmap(q_params)

    sfh = sfh_galpop(
        tarr,
        mah_params,
        u_ms_params,
        u_q_params,
        LGT0,
        FB,
        ms_param_type="unbounded",
        q_param_type="unbounded",
    )

    assert np.allclose(sfh, u_sfh, rtol=1e-3)


def test_fb_value_propagates_to_sfh_singlegal():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB)
    x = 2.0
    sfh2 = sfh_singlegal(tarr, mah_params, ms_params, q_params, LGT0, FB * x)
    assert np.allclose(sfh2, x * sfh)


def test_fb_value_propagates_to_sfh_galpop():
    lgt0, mah_params, ms_params, q_params = _get_all_default_params()

    mah_params = np.array(mah_params).reshape((1, -1))
    ms_params = np.array(ms_params).reshape((1, -1))
    q_params = np.array(q_params).reshape((1, -1))

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, LGT0, FB)
    x = 2.0
    sfh2 = sfh_galpop(tarr, mah_params, ms_params, q_params, LGT0, FB * x)
    assert np.allclose(sfh2, x * sfh)


def test_sfh_galpop_is_always_strictly_positive():
    n_gals, n_t = 1_000, 100
    lgt0, mah_params, __, __ = _get_all_default_params()
    tarr = np.linspace(0.1, 10**lgt0, n_t)

    mah_params_singlegal = np.array((12.0, 0.0, 3.0, 2.0))
    mah_params_galpop = np.tile(mah_params_singlegal, n_gals).reshape((n_gals, 4))
    u_ms_params_galpop = np.random.uniform(-100, 100, size=(n_gals, 5))
    u_q_params_galpop = np.random.uniform(-100, 100, size=(n_gals, 4))

    sfh = sfh_galpop(
        tarr,
        mah_params_galpop,
        u_ms_params_galpop,
        u_q_params_galpop,
        LGT0,
        FB,
        ms_param_type="unbounded",
        q_param_type="unbounded",
    )
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
