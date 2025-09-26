""" """

import numpy as np
import jax.numpy as jnp

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS

# === Adjust these imports to your real module paths ===
from ...defaults import SFR_MIN
from ..main_sequence_kernels_mgash import DEFAULT_MS_PARAMS
from ..quenching_kernels import (
    DEFAULT_Q_PARAMS,
    DEFAULT_Q_PARAMS_UNQUENCHED,
)
from ..history_kernel_builders_mgash import (
    _sfh_singlegal_scalar,
    _sfh_singlegal_kern,
    _sfh_galpop_kern,
)

# =====================================================


def _tarr():
    return np.linspace(0.1, 13.8, 200)


def _logt0():
    return jnp.log10(13.8)


# -----------------------
# Scalar vs vectorization
# -----------------------


def test_sfh_singlegal_kern_matches_scalar_calls_unquenched():
    """_sfh_singlegal_kern (vmapped over tform) should match stacking scalar evaluations."""
    tarr = _tarr()
    logt0 = _logt0()
    fb = 0.16

    # vmapped over time
    sfh_vec = _sfh_singlegal_kern(
        tarr,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS_UNQUENCHED,
        logt0,
        fb,
    )

    # scalar calls stacked
    sfh_stack = jnp.array(
        [
            _sfh_singlegal_scalar(
                t,
                DEFAULT_MAH_PARAMS,
                DEFAULT_MS_PARAMS,
                DEFAULT_Q_PARAMS_UNQUENCHED,
                logt0,
                fb,
            )
            for t in tarr
        ]
    )

    assert jnp.all(jnp.isfinite(sfh_vec))
    assert jnp.all(sfh_vec >= 0.0)
    assert np.allclose(np.array(sfh_vec), np.array(sfh_stack), rtol=1e-6, atol=1e-9)


def test_sfh_singlegal_kern_linear_in_fb_when_not_floored():
    """For elements not hitting the SFR floor, doubling fb should double SFR."""
    tarr = _tarr()
    logt0 = _logt0()
    fb1, fb2 = 0.16, 0.32  # 2x

    sfh1 = _sfh_singlegal_kern(
        tarr,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS_UNQUENCHED,
        logt0,
        fb1,
    )
    sfh2 = _sfh_singlegal_kern(
        tarr,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS_UNQUENCHED,
        logt0,
        fb2,
    )

    # Only compare where both are safely above the floor
    mask = (sfh1 > SFR_MIN * 1.5) & (sfh2 > SFR_MIN * 1.5)
    if jnp.any(mask):
        assert np.allclose(
            np.array(sfh2[mask]), 2.0 * np.array(sfh1[mask]), rtol=1e-6, atol=1e-9
        )
    else:
        # Degenerate case: everything floored; still assert they equal the floor.
        assert np.allclose(np.array(sfh1), SFR_MIN)
        assert np.allclose(np.array(sfh2), SFR_MIN)


# -----------------------
# Floor behavior
# -----------------------


def test_sfh_floor_applied_when_expected():
    """With tiny fb (and quenching), SFR should be clamped to SFR_MIN."""
    tarr = _tarr()
    logt0 = _logt0()
    fb_tiny = 1e-12

    # Using default quenching is fine; tiny fb should ensure we hit the floor
    sfh = _sfh_singlegal_kern(
        tarr, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS, logt0, fb_tiny
    )

    assert jnp.all(jnp.isfinite(sfh))
    assert np.allclose(np.array(sfh), SFR_MIN)  # everything at the floor


# -----------------------
# Galaxy-population wrapper
# -----------------------


def test_sfh_galpop_kern_shapes_and_consistency():
    """
    _sfh_galpop_kern vmaps over galaxies: (t, mah_params[i], ms_params[i], q_params[i]).
    Verify shape, finiteness, non-negativity, and agreement with per-galaxy calls.
    """
    tarr = _tarr()
    logt0 = _logt0()
    fb = 0.17

    # Build a small population of 3 galaxies with slightly perturbed params
    # DEFAULT_MAH_PARAMS is a namedtuple; we pass arrays of sequences into the wrapper,
    # which internally converts each row to the namedtuple via _make.
    mah_defaults = jnp.array(DEFAULT_MAH_PARAMS)
    mah_batch = jnp.stack(
        [
            mah_defaults,
            mah_defaults * (1.0 + 1e-3),
            mah_defaults * (1.0 - 1e-3),
        ],
        axis=0,
    )

    ms_defaults = jnp.array(DEFAULT_MS_PARAMS)
    ms_batch = jnp.stack(
        [
            ms_defaults,
            ms_defaults + jnp.array([0.05, -0.05, 0.1, -0.1]),
            ms_defaults + jnp.array([-0.05, 0.05, -0.1, 0.1]),
        ],
        axis=0,
    )

    q_defaults = jnp.array(DEFAULT_Q_PARAMS)
    q_batch = jnp.stack(
        [
            q_defaults,
            q_defaults + jnp.array([0.02, 0.02, -0.02, 0.0]),
            q_defaults + jnp.array([-0.02, -0.02, 0.02, 0.0]),
        ],
        axis=0,
    )

    # Vectorized population call
    sfh_pop = _sfh_galpop_kern(tarr, mah_batch, ms_batch, q_batch, logt0, fb)
    # Expected shape: (ngal, nt)
    assert sfh_pop.shape == (3, tarr.shape[0])
    assert jnp.all(jnp.isfinite(sfh_pop))
    assert jnp.all(sfh_pop >= 0.0)

    # Compare with per-galaxy calls of the time-vmap kernel
    # Note: _sfh_singlegal_kern needs a namedtuple for mah_params (not arrays),
    # but _sfh_galpop_kern handles that conversion internally. So we mimic that here.
    per_gal = []
    for i in range(3):
        mah_nt = DEFAULT_MAH_PARAMS._make(np.array(mah_batch[i]))
        per_gal.append(
            _sfh_singlegal_kern(
                tarr,
                mah_nt,
                tuple(np.array(ms_batch[i])),
                tuple(np.array(q_batch[i])),
                logt0,
                fb,
            )
        )
    sfh_stack = jnp.stack(per_gal, axis=0)

    assert np.allclose(np.array(sfh_pop), np.array(sfh_stack), rtol=1e-6, atol=1e-9)


# -----------------------
# Basic sanity with defaults
# -----------------------


def test_sfh_defaults_basic_sanity_unquenched():
    """Smoke test with defaults & no quenching: finite, non-negative, not identically zero."""
    tarr = _tarr()
    logt0 = _logt0()
    fb = 0.16

    sfh = _sfh_singlegal_kern(
        tarr,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS_UNQUENCHED,
        logt0,
        fb,
    )

    assert jnp.all(jnp.isfinite(sfh))
    assert jnp.all(sfh >= 0.0)
    assert jnp.any(sfh > SFR_MIN)
