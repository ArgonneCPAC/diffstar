"""
"""
import numpy as np

from .. import sfh_galpop, sfh_singlegal
from ..defaults import SFR_MIN
from ..kernel_builders import get_sfh_from_mah_kern
from .test_diffstar_is_frozen import _get_default_mah_params, _get_default_sfr_u_params


def _get_all_default_params():
    u_ms_params, u_q_params = _get_default_sfr_u_params()
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return lgt0, mah_params, u_ms_params, u_q_params


def test_sfh_singlegal_evaluates():
    lgt0, mah_params, u_ms_params, u_q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, u_ms_params, u_q_params)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh >= SFR_MIN)
    assert sfh.shape == (n_t,)


def test_sfh_singlegal_agrees_with_kernel_builder():
    kern = get_sfh_from_mah_kern(tobs_loop="scan")

    lgt0, mah_params, u_ms_params, u_q_params = _get_all_default_params()

    n_t = 100
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_singlegal(tarr, mah_params, u_ms_params, u_q_params)
    sfh2 = kern(tarr, mah_params, u_ms_params, u_q_params)

    assert np.allclose(sfh, sfh2, rtol=1e-4)


def test_sfh_galpop_evaluates():
    n_t = 100
    lgt0, mah_params, u_ms_params, u_q_params = _get_all_default_params()
    mah_params = np.array(mah_params).reshape((1, -1))
    u_ms_params = np.array(u_ms_params).reshape((1, -1))
    u_q_params = np.array(u_q_params).reshape((1, -1))
    tarr = np.linspace(0.1, 10**lgt0, n_t)
    sfh = sfh_galpop(tarr, mah_params, u_ms_params, u_q_params)
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
    sfh = sfh_galpop(tarr, mah_params, u_ms_params, u_q_params)
    sfh2 = kern(tarr, mah_params, u_ms_params, u_q_params)

    assert np.allclose(sfh, sfh2, rtol=1e-4)
