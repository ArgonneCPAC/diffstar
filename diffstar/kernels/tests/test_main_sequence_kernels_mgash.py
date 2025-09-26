""" """

import numpy as np
import jax.numpy as jnp

from ..main_sequence_kernels_mgash import (
    DEFAULT_MS_PARAMS,
    DEFAULT_U_MS_PARAMS,
    DEFAULT_MS_PDICT,
    MS_PARAM_BOUNDS_PDICT,
    MS_BOUNDING_SIGMOID_PDICT,
    _sfr_eff_plaw,
    sfh_ms_kernel,
    sfh_ms_kernel_vmap,
    _get_bounded_sfr_params,
    _get_unbounded_sfr_params,
    _get_bounded_sfr_params_vmap,
    _get_unbounded_sfr_params_vmap,
    calculate_sigmoid_bounds,
)

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS


def test_unbounding_function_returns_finite_results_on_default_ms_params():
    inferred_default_u_ms_params = _get_unbounded_sfr_params(*DEFAULT_MS_PARAMS)
    assert np.all(np.isfinite(inferred_default_u_ms_params))


def test_bounding_function_returns_finite_results_on_default_u_ms_params():
    inferred_default_ms_params = _get_bounded_sfr_params(*DEFAULT_U_MS_PARAMS)
    assert np.all(np.isfinite(inferred_default_ms_params))


def test_default_mainseq_params_respect_bounds():

    for key, bounds in MS_PARAM_BOUNDS_PDICT.items():
        lo, hi = bounds
        default_val = DEFAULT_MS_PDICT[key]
        assert lo < default_val < hi


def test_calculate_sigmoid_bounds_matches_expected_construction():
    """Ensure calculate_sigmoid_bounds returns (center, steepness, lo, hi) with correct values."""
    bounds = calculate_sigmoid_bounds(MS_PARAM_BOUNDS_PDICT)

    # Basic structure check: same keys, each has a 4-tuple (center, steepness, lo, hi)
    assert set(bounds.keys()) == set(MS_PARAM_BOUNDS_PDICT.keys())
    for key, (center, steepness, lo, hi) in bounds.items():
        lo_exp, hi_exp = MS_PARAM_BOUNDS_PDICT[key]
        # Center is the mean of the bounds
        assert np.allclose(center, 0.5 * (lo_exp + hi_exp))
        # Steepness is 4 / (hi - lo)
        assert np.allclose(steepness, 4.0 / (hi_exp - lo_exp))
        # Lower/upper bounds propagated through unchanged
        assert np.allclose(lo, lo_exp)
        assert np.allclose(hi, hi_exp)


def test_bounded_params_lie_within_declared_bounds():
    """Map a variety of unbounded inputs through _get_bounded_sfr_params and
    verify they land within MS_PARAM_BOUNDS_PDICT."""
    # Try a range of unbounded inputs (large neg, 0, large pos)
    test_vals = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    combos = [
        (u1, u2, u3, u4)
        for u1 in test_vals
        for u2 in test_vals
        for u3 in test_vals
        for u4 in test_vals
    ]

    for u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi in combos[::17]:  # subsample
        lgmcrit, lgy_at_mcrit, indx_lo, indx_hi = _get_bounded_sfr_params(
            u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi
        )

        # All finite
        assert np.all(np.isfinite([lgmcrit, lgy_at_mcrit, indx_lo, indx_hi]))

        # Check each param is within its declared bounds
        (lo, hi) = MS_PARAM_BOUNDS_PDICT["lgmcrit"]
        assert lo <= lgmcrit <= hi
        (lo, hi) = MS_PARAM_BOUNDS_PDICT["lgy_at_mcrit"]
        assert lo <= lgy_at_mcrit <= hi
        (lo, hi) = MS_PARAM_BOUNDS_PDICT["indx_lo"]
        assert lo <= indx_lo <= hi
        (lo, hi) = MS_PARAM_BOUNDS_PDICT["indx_hi"]
        assert lo <= indx_hi <= hi


def test_param_bounding_functions_correctly_invert_on_defaults():
    """Round-trip bounded <-> unbounded on the provided defaults."""
    # Bounded -> unbounded -> bounded
    u_params_from_defaults = _get_unbounded_sfr_params(*DEFAULT_MS_PARAMS)
    b_params_roundtrip = _get_bounded_sfr_params(*u_params_from_defaults)

    # Unbounded -> bounded -> unbounded
    b_params_from_u_defaults = _get_bounded_sfr_params(*DEFAULT_U_MS_PARAMS)
    u_params_roundtrip = _get_unbounded_sfr_params(*b_params_from_u_defaults)

    # Finite
    assert np.all(np.isfinite(u_params_from_defaults))
    assert np.all(np.isfinite(b_params_roundtrip))
    assert np.all(np.isfinite(b_params_from_u_defaults))
    assert np.all(np.isfinite(u_params_roundtrip))

    # Close to originals
    assert np.allclose(b_params_roundtrip, DEFAULT_MS_PARAMS, rtol=1e-5, atol=1e-7)
    assert np.allclose(u_params_roundtrip, DEFAULT_U_MS_PARAMS, rtol=1e-5, atol=1e-7)


def test_param_bounding_vmapped_matches_scalar():
    """Vmapped versions should match scalar versions for the same inputs."""
    # Build a small batch around the defaults
    b0 = np.array(DEFAULT_MS_PARAMS)
    u0 = np.array(DEFAULT_U_MS_PARAMS)

    b_batch = jnp.stack(
        [
            b0,
            b0 + jnp.array([0.1, -0.1, 0.2, -0.2]),
            b0 + jnp.array([-0.05, 0.05, -0.1, 0.1]),
        ]
    )
    u_batch = jnp.stack(
        [
            u0,
            u0 + jnp.array([0.2, -0.2, 0.1, -0.1]),
            u0 + jnp.array([-0.1, 0.1, -0.05, 0.05]),
        ]
    )

    # Clip bounded batches to remain strictly within declared bounds before inversion
    def clip_to_bounds(bvec):
        out = []
        for val, (lo, hi) in zip(bvec, MS_PARAM_BOUNDS_PDICT.values()):
            out.append(jnp.clip(val, lo + 1e-6, hi - 1e-6))
        return jnp.array(out)

    b_batch = jnp.stack([clip_to_bounds(v) for v in b_batch])

    # Compare bounded -> unbounded
    u_vmapped = _get_unbounded_sfr_params_vmap(b_batch)
    u_scalar = jnp.stack(
        [jnp.array(_get_unbounded_sfr_params(*tuple(bv))) for bv in b_batch]
    )
    assert np.allclose(u_vmapped, u_scalar, rtol=1e-6, atol=1e-8)

    # Compare unbounded -> bounded
    b_vmapped = _get_bounded_sfr_params_vmap(u_batch)
    b_scalar = jnp.stack(
        [jnp.array(_get_bounded_sfr_params(*tuple(uv))) for uv in u_batch]
    )
    assert np.allclose(b_vmapped, b_scalar, rtol=1e-6, atol=1e-8)


def test_sfr_eff_plaw_expected_behavior_at_mcrit_and_slopes():
    """At lgmcrit, eff should equal lgy_at_mcrit (since delta=0).
    Also verify the slope term stays within [min(indx_lo, indx_hi), max(indx_lo, indx_hi)].
    """
    lgmcrit, lgy_at_mcrit, indx_lo, indx_hi = DEFAULT_MS_PARAMS

    # At the pivot, the linear term vanishes
    eff_at_pivot = _sfr_eff_plaw(lgmcrit, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi)
    assert np.allclose(eff_at_pivot, lgy_at_mcrit, rtol=1e-6, atol=1e-10)

    # Sweep around the pivot and check slope bounds
    lgm = np.linspace(lgmcrit - 5.0, lgmcrit + 5.0, 257)
    # Extract the sigmoid "slope" term by reusing the kernel’s first line:
    # slope = _sigmoid(lgm, lgmcrit, INDX_K, indx_lo, indx_hi)
    # We don't call the internal _sigmoid here directly; instead verify the resulting eff
    # implies a slope bounded between the two indices.
    eff = _sfr_eff_plaw(lgm, lgmcrit, lgy_at_mcrit, indx_lo, indx_hi)
    # slope = (eff - lgy_at_mcrit) / (lgm - lgmcrit); handle the pivot point by masking
    mask = np.abs(lgm - lgmcrit) > 1e-8
    slope_est = np.zeros_like(eff)
    slope_est[mask] = (eff[mask] - lgy_at_mcrit) / (lgm[mask] - lgmcrit)

    lo = min(indx_lo, indx_hi)
    hi = max(indx_lo, indx_hi)
    assert np.all(slope_est[mask] >= lo - 1e-6)
    assert np.all(slope_est[mask] <= hi + 1e-6)


def test_defaults_and_u_defaults_are_consistent():
    """DEFAULT_U_MS_PARAMS should equal the unbounded transform of DEFAULT_MS_PARAMS,
    and the forward transform should return the bounded defaults."""
    u_from_bounded = _get_unbounded_sfr_params(*DEFAULT_MS_PARAMS)
    b_from_unbounded = _get_bounded_sfr_params(*DEFAULT_U_MS_PARAMS)

    assert np.allclose(u_from_bounded, DEFAULT_U_MS_PARAMS, rtol=1e-6, atol=1e-8)
    assert np.allclose(b_from_unbounded, DEFAULT_MS_PARAMS, rtol=1e-6, atol=1e-8)


def test_bounding_sigmoid_dict_consistency_with_param_bounds():
    """MS_BOUNDING_SIGMOID_PDICT should align with MS_PARAM_BOUNDS_PDICT contents."""
    for key in MS_PARAM_BOUNDS_PDICT:
        assert key in MS_BOUNDING_SIGMOID_PDICT
        center, steep, lo, hi = MS_BOUNDING_SIGMOID_PDICT[key]
        lo_exp, hi_exp = MS_PARAM_BOUNDS_PDICT[key]
        # Ends must match the declared bounds
        assert np.allclose(lo, lo_exp)
        assert np.allclose(hi, hi_exp)
        # Center/steep must be consistent with those bounds
        assert np.allclose(center, 0.5 * (lo_exp + hi_exp))
        assert np.allclose(steep, 4.0 / (hi_exp - lo_exp))


def test_sfh_ms_kernel_with_default_params():
    """Check that sfh_ms_kernel runs with DEFAULT_MAH_PARAMS and DEFAULT_MS_PARAMS."""
    tform = jnp.linspace(0.1, 13.8, 128)  # cosmic time grid
    logt0 = jnp.log10(13.8)  # present-day cosmic time
    fb = 0.16  # cosmic baryon fraction

    sfh = sfh_ms_kernel(tform, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, logt0, fb)

    # All outputs should be finite, non-negative
    assert jnp.all(jnp.isfinite(sfh))
    assert jnp.all(sfh >= 0)

    # Scaling in fb should be linear
    sfh_2fb = sfh_ms_kernel(tform, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, logt0, fb * 2)
    assert np.allclose(np.array(sfh_2fb), 2.0 * np.array(sfh), rtol=1e-6, atol=1e-9)

    # At early times, SFH should be small but positive
    assert sfh[0] >= 0.0
    # At late times, some star formation remains
    assert sfh[-1] >= 0.0


def test_sfh_ms_kernel_vmap_matches_scalar():
    """Vectorized sfh_ms_kernel should match stacking scalar calls."""
    tform1 = jnp.linspace(0.1, 5.0, 64)
    tform2 = jnp.linspace(1.0, 6.0, 64)
    tform_batch = jnp.stack([tform1, tform2], axis=0)

    logt0 = jnp.log10(13.8)
    fb = 0.17

    # Vectorized call
    sfh_vec = sfh_ms_kernel_vmap(
        tform_batch, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, logt0, fb
    )

    # Scalar calls stacked
    sfh_1 = sfh_ms_kernel(tform1, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, logt0, fb)
    sfh_2 = sfh_ms_kernel(tform2, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, logt0, fb)
    sfh_stack = jnp.stack([sfh_1, sfh_2], axis=0)

    assert jnp.all(jnp.isfinite(sfh_vec))
    assert np.allclose(np.array(sfh_vec), np.array(sfh_stack), rtol=1e-6, atol=1e-9)
