"""
"""
import numpy as np
from jax import jit as jjit
from jax import vmap

from ..defaults import DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS, FB, LGT0
from ..kernels.kernel_builders import get_sfh_from_mah_kern
from .test_gas import _get_default_mah_params


def _get_all_default_params():
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index
    return mah_params, u_ms_params, u_q_params


def test_get_sfh_kernel_agrees_with_vmap_tacc(n_t=400, n_steps=100):
    """Enforce that when looping over tacc, vmap vs scan results agree within 5%"""
    u_ms_params, u_q_params = DEFAULT_U_MS_PARAMS, DEFAULT_U_Q_PARAMS
    tarr = np.linspace(0.1, 10**LGT0, 100)

    all_mah_params = _get_default_mah_params()
    lgt0, logmp, mah_logtc, k, early_index, late_index = all_mah_params
    mah_params = logmp, mah_logtc, early_index, late_index

    sfh_from_mah_kern_scan = get_sfh_from_mah_kern(n_steps=n_steps, tobs_loop="scan")
    sfh_from_mah_kern = get_sfh_from_mah_kern(n_steps=n_steps)
    sfh_from_mah_vmap = jjit(
        vmap(sfh_from_mah_kern, in_axes=[0, None, None, None, None, None])
    )
    vmap_sfh = sfh_from_mah_vmap(tarr, mah_params, u_ms_params, u_q_params, LGT0, FB)
    lax_sfh = sfh_from_mah_kern_scan(
        tarr, mah_params, u_ms_params, u_q_params, LGT0, FB
    )

    assert np.allclose(vmap_sfh, lax_sfh, rtol=0.05)


def test_sfh_kernel_builder_tobs_loops_are_self_consistent():
    """Enforce that when looping over tobs, results agree with pure python loop
    regardless of whether scan or vmap is used
    """
    mah_params, u_ms_params, u_q_params = _get_all_default_params()
    default_params = mah_params, u_ms_params, u_q_params, LGT0, FB

    n_tobs = 10
    tarr = np.linspace(0.1, 13.7, n_tobs)

    sfh_scalar_kern = get_sfh_from_mah_kern()
    sfr_at_t0 = sfh_scalar_kern(tarr[0], *default_params)
    assert sfr_at_t0.shape == ()

    sfh_python_loop = [sfh_scalar_kern(t, *default_params) for t in tarr]

    sfh_vmap_tobs_kern = get_sfh_from_mah_kern(tobs_loop="vmap")
    sfh_vmap_tobs = sfh_vmap_tobs_kern(tarr, *default_params)
    assert sfh_vmap_tobs.shape == (n_tobs,)
    assert np.allclose(sfh_python_loop, sfh_vmap_tobs, rtol=1e-4)

    sfh_scan_tobs_kern = get_sfh_from_mah_kern(tobs_loop="scan")
    sfh_scan_tobs = sfh_scan_tobs_kern(tarr, *default_params)
    assert np.allclose(sfh_python_loop, sfh_scan_tobs, rtol=1e-4)


def test_sfh_kernel_builder_galpop_loops_are_self_consistent():
    """Enforce that when looping over galpop, results agree with pure python loop
    regardless of whether scan or vmap is used
    """
    mah_params, u_ms_params, u_q_params = _get_all_default_params()
    default_params = mah_params, u_ms_params, u_q_params, LGT0, FB
    n_mah = len(mah_params)
    n_ms = len(u_ms_params)
    n_q = len(u_q_params)

    tobs = 5.0

    sfh_scalar_kern = get_sfh_from_mah_kern()
    sfr_at_tobs = sfh_scalar_kern(tobs, *default_params)

    n_galpop = 3
    outshape = (n_galpop,)
    mah_params_galpop = np.tile(mah_params, n_galpop).reshape((n_galpop, n_mah))
    u_ms_params_galpop = np.tile(u_ms_params, n_galpop).reshape((n_galpop, n_ms))
    u_q_params_galpop = np.tile(u_q_params, n_galpop).reshape((n_galpop, n_q))
    galpop_args = mah_params_galpop, u_ms_params_galpop, u_q_params_galpop, LGT0, FB

    sfr_at_tobs_galpop_python_loop = np.tile(sfr_at_tobs, n_galpop).reshape(outshape)

    sfr_vmap_galpop_kern = get_sfh_from_mah_kern(galpop_loop="vmap")
    sfr_at_tobs_vmap_galpop = sfr_vmap_galpop_kern(tobs, *galpop_args)
    assert sfr_at_tobs_vmap_galpop.shape == outshape
    assert np.allclose(
        sfr_at_tobs_vmap_galpop, sfr_at_tobs_galpop_python_loop, rtol=1e-4
    )

    sfr_scan_galpop_kern = get_sfh_from_mah_kern(galpop_loop="scan")
    sfr_at_tobs_scan_galpop = sfr_scan_galpop_kern(tobs, *galpop_args)
    assert np.allclose(
        sfr_at_tobs_scan_galpop, sfr_at_tobs_galpop_python_loop, rtol=1e-4
    )


def test_main_sequence_kernel_builder_tobs_and_galpop_loops_are_self_consistent():
    """Enforce that when looping over both tobs and galpop, results are independent of
    whether scan or vmap is used
    """
    mah_params, u_ms_params, u_q_params = _get_all_default_params()
    default_params = mah_params, u_ms_params, u_q_params, LGT0, FB
    n_mah = len(mah_params)
    n_ms = len(u_ms_params)
    n_q = len(u_q_params)

    n_tobs = 10
    tarr = np.linspace(0.1, 13.7, n_tobs)

    n_galpop = 3
    outshape = (n_galpop, n_tobs)
    mah_params_galpop = np.tile(mah_params, n_galpop).reshape((n_galpop, n_mah))
    u_ms_params_galpop = np.tile(u_ms_params, n_galpop).reshape((n_galpop, n_ms))
    u_q_params_galpop = np.tile(u_q_params, n_galpop).reshape((n_galpop, n_q))
    galpop_args = mah_params_galpop, u_ms_params_galpop, u_q_params_galpop, LGT0, FB

    sfh_scalar_kern = get_sfh_from_mah_kern()
    sfh_python_loop = [sfh_scalar_kern(t, *default_params) for t in tarr]

    sfh_python_loops = np.tile(sfh_python_loop, n_galpop).reshape(outshape)

    sfh_vmap_tobs_vmap_galpop_func = get_sfh_from_mah_kern(
        tobs_loop="vmap", galpop_loop="vmap"
    )
    sfh_vmap_vmap = sfh_vmap_tobs_vmap_galpop_func(tarr, *galpop_args)
    assert sfh_vmap_vmap.shape == outshape
    assert np.allclose(sfh_vmap_vmap, sfh_python_loops, rtol=1e-4)

    sfh_vmap_tobs_scan_galpop_func = get_sfh_from_mah_kern(
        tobs_loop="vmap", galpop_loop="scan"
    )
    sfh_vmap_scan = sfh_vmap_tobs_scan_galpop_func(tarr, *galpop_args)
    assert np.allclose(sfh_vmap_scan, sfh_python_loops, rtol=1e-4)

    sfh_scan_tobs_vmap_galpop_func = get_sfh_from_mah_kern(
        tobs_loop="scan", galpop_loop="vmap"
    )
    sfh_scan_vmap = sfh_scan_tobs_vmap_galpop_func(tarr, *galpop_args)
    assert np.allclose(sfh_scan_vmap, sfh_python_loops, rtol=1e-4)

    sfh_scan_tobs_scan_galpop_func = get_sfh_from_mah_kern(
        tobs_loop="scan", galpop_loop="scan"
    )
    sfh_scan_scan = sfh_scan_tobs_scan_galpop_func(tarr, *galpop_args)
    assert np.allclose(sfh_scan_scan, sfh_python_loops, rtol=1e-4)
