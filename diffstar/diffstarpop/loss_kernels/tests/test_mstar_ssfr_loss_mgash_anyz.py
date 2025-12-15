import numpy as np
import jax.numpy as jnp
import pytest

from ...kernels.defaults_mgash import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)

from .. import mstar_ssfr_loss_mgash_anyz as mod
from .load_loss_data import load_loss_data_h5
from pathlib import Path

DATA_DIR = Path(__file__).parent / "testing_data"
H5_PATH = DATA_DIR / "loss_kernels_testing_data_10halos.h5"


@pytest.fixture(scope="session")
def flat_u_params():
    return jnp.asarray(DEFAULT_DIFFSTARPOP_U_PARAMS)


def _require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing.\n")


@pytest.fixture(scope="session")
def h5_path() -> Path:
    _require_file(H5_PATH)
    return H5_PATH


@pytest.fixture(scope="session")
def loss_data_all(h5_path):
    # returns (loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat)
    return load_loss_data_h5(str(h5_path))


@pytest.fixture(scope="session")
def loss_data_mstar(loss_data_all):
    mstar, _, _ = loss_data_all
    return mstar


@pytest.fixture(scope="session")
def loss_data_ssfr(loss_data_all):
    _, ssfr, _ = loss_data_all
    return ssfr


@pytest.fixture(scope="session")
def loss_data_ssfr_sat(loss_data_all):
    _, _, ssfr_sat = loss_data_all
    return ssfr_sat


# -------------------------------
# Unit tests
# -------------------------------


def test_compute_diff_histograms_mstar_atmobs_z_normalization_and_shape():
    # Small synthetic flattened inputs
    logmstar_bins = jnp.linspace(8.0, 11.0, 7)  # 6 bins
    n_halos = 15
    n_samples = 9
    # Build a set of values spanning the bins (flattened inside the function)
    rng = np.linspace(8.05, 10.95, n_halos * n_samples).reshape(n_halos, n_samples)
    weights = np.ones_like(rng)

    pdf = mod.compute_diff_histograms_mstar_atmobs_z_vmap(
        logmstar_bins,
        jnp.array(rng),
        jnp.array(weights),
    )
    # Expect one probability per bin (len(bins)-1)
    assert pdf.shape == (n_halos, len(logmstar_bins) - 1)
    np.testing.assert_allclose(
        np.sum(np.asarray(pdf), axis=1), np.ones(n_halos), atol=1e-6
    )


def test_h5_data_shapes_and_sanity(loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat):
    # Unpack mstar tuple
    (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        lgt0,
        fb,
        logmstar_bins_pdf,
        mstar_counts_target,
    ) = loss_data_mstar

    # Basic expected shapes
    assert mah_params_data.ndim == 3 and mah_params_data.shape[1] == 5
    n_obs = mah_params_data.shape[0]
    n_halo = mah_params_data.shape[2]
    assert n_obs == 52
    assert n_halo == logmp0_data.shape[1] == upid_data.shape[1] == 10
    assert logmp0_data.shape == (n_obs, n_halo)
    assert upid_data.shape == (n_obs, n_halo)
    assert lgmu_infall_data.shape == (n_obs, n_halo)
    assert logmhost_infall_data.shape == (n_obs, n_halo)
    assert gyr_since_infall_data.shape == (n_obs, n_halo)
    assert ran_key_data.shape == (n_obs, 2)
    assert t_obs_targets.shape == (n_obs,)
    # Bins/targets are implementation-specific; just sanity-check dims
    assert logmstar_bins_pdf.ndim == 1
    assert mstar_counts_target.ndim == 2 and mstar_counts_target.shape[0] == n_obs

    # SSFR (centrals)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins_pdf2,
        logssfr_bins_pdf,
        mhalo_pdf_cen_ragged,
        indx_pdf,
        target_mstar_ids,
        target_data,
    ) = loss_data_ssfr

    assert ndbins_lo.shape == ndbins_hi.shape
    assert logmstar_bins_pdf2.ndim == 1
    assert logssfr_bins_pdf.ndim == 1
    assert indx_pdf.shape == mhalo_pdf_cen_ragged.shape
    assert target_mstar_ids.ndim == 1

    # SSFR (satellites)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        ndbins_lo_s,
        ndbins_hi_s,
        logmstar_bins_pdf_s,
        logssfr_bins_pdf_s,
        mhalo_pdf_sat_ragged,
        indx_pdf_s,
        target_mstar_ids_s,
        target_data_sat,
    ) = loss_data_ssfr_sat

    assert ndbins_lo_s.shape == ndbins_hi_s.shape == ndbins_lo.shape
    assert logmstar_bins_pdf_s.shape == logmstar_bins_pdf.shape
    assert logssfr_bins_pdf_s.shape == logssfr_bins_pdf.shape
    assert indx_pdf_s.shape == mhalo_pdf_sat_ragged.shape
    assert target_mstar_ids_s.shape == target_mstar_ids.shape


def test_mstar_wrappers_predict_and_loss_zero_when_target_equals_pred(
    flat_u_params, loss_data_mstar
):

    # Predictions
    preds = mod.get_pred_mstar_data_wrapper(flat_u_params, loss_data_mstar)
    assert preds.shape == loss_data_mstar[-1].shape  # match mstar_counts_target
    # Each observation should be a PDF over bins
    row_sums = np.asarray(jnp.sum(preds, axis=1))
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    # Loss + grads on original target
    loss_val, grads = mod.loss_mstar_kern_tobs_grad_wrapper(
        flat_u_params, loss_data_mstar
    )
    assert np.isfinite(loss_val)
    assert grads.shape == flat_u_params.shape
    assert np.all(np.isfinite(np.asarray(grads)))

    # Now set the target equal to prediction -> loss should be ~0
    mstar_loss_data_eq = loss_data_mstar[:-1] + (preds,)
    loss_val_eq, _ = mod.loss_mstar_kern_tobs_grad_wrapper(
        flat_u_params, mstar_loss_data_eq
    )
    assert float(loss_val_eq) < 1e-5


def test_combined_loss_wrappers_run_and_return_finite_values(
    flat_u_params, loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat
):

    # 2-loss combined (M★ + sSFR centrals)
    loss2, grads2 = mod.loss_combined_wrapper(
        flat_u_params, loss_data_mstar, loss_data_ssfr
    )
    assert np.isfinite(loss2)
    assert grads2.shape == flat_u_params.shape
    assert np.all(np.isfinite(np.asarray(grads2)))

    # 3-loss combined (M★ + sSFR centrals + sSFR satellites)
    loss3, grads3 = mod.loss_combined_3loss_wrapper(
        flat_u_params, loss_data_mstar, loss_data_ssfr, loss_data_ssfr_sat
    )
    assert np.isfinite(loss3)
    assert grads3.shape == flat_u_params.shape
    assert np.all(np.isfinite(np.asarray(grads3)))
