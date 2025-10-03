""" """

import os
import h5py
import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop
from diffstar.defaults import LGT0
from jax import random as jran
from jax import numpy as jnp


def get_loss_data_smhm(indir, nhalos):
    # Load SMHM data ---------------------------------------------
    print("Loading SMHM data...")

    with h5py.File(indir + "smdpl_smhm.h5", "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        # smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        """
            hdfout["counts_diff"] = wcounts
            hdfout["hist_diff"] = whist
            hdfout["counts"] = counts
            hdfout["hist"] = hist
            hdfout["smhm_diff"] = whist / wcounts
            hdfout["smhm"] = hist / counts
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["subvol_used"] = subvol_used
        """

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    with h5py.File(indir + "smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        # logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        # ms_params_samp = hdf["ms_params_samp"][:]
        # q_params_samp = hdf["q_params_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        # tobs_val = hdf["tobs_val"][:]
        # redshift_val = hdf["redshift_val"][:]

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    smhm_targets = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < nhalos:
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=False)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            smhm_targets.append(smhm[i, j])
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    smhm_targets = np.array(smhm_targets)

    ran_key_data = jran.split(ran_key, len(smhm_targets))
    loss_data = (
        mah_params_data,
        logmp0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        smhm_targets,
    )

    plot_data = (
        age_targets,
        logmh_binsc,
        tobs_id,
        logmh_id,
        tarr_logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        redshift_targets,
        smhm,
        mah_params_samp,
        upid_samp,
    )

    return loss_data, plot_data


def get_loss_data_pdfs_mstar(indir, nhalos):
    # Load PDF data ---------------------------------------------
    print("Loading PDF Mstar data...")

    fname = os.path.join(indir, "smdpl_smhm.h5")
    with h5py.File(fname, "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    fname = os.path.join(indir, "smdpl_smhm_samples_haloes.h5")
    with h5py.File(fname, "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    fname = os.path.join(indir, "smdpl_mstar_ssfr.h5")
    with h5py.File(fname, "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(mstar_wcounts[i, j] / mstar_wcounts[i, j].sum())
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000
    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(mstar_wcounts[i, j] / mstar_wcounts[i, j].sum())
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar_pred = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )
    plot_data = (
        logmstar_bins_pdf,
        mstar_wcounts,
        age_targets,
        redshift_targets,
        tobs_id,
        logmh_id,
        logmh_binsc,
        loss_data_mstar_pred,
    )

    return loss_data_mstar, plot_data


def prepare_ragged(indx_pdf, nmhalo_pdf, index_mhalo):
    """Run this outside jit once per dataset to create dense, shape-stable arrays."""
    nz = len(indx_pdf)
    Mmax = max(len(ix) for ix in indx_pdf)

    # Build dense (nz, Mmax) arrays for indices, weights, and mask
    idx_np = np.zeros((nz, Mmax), dtype=jnp.int32)
    w_np = np.zeros((nz, Mmax), dtype=nmhalo_pdf.dtype)
    # msk_np = np.zeros((nz, Mmax), dtype=bool)

    for z in range(nz):
        m = len(indx_pdf[z])
        idx_np[z, :m] = jnp.asarray(indx_pdf[z])
        w_np[z, :m] = jnp.asarray(nmhalo_pdf[z, index_mhalo[z]])
        # msk_np[z, :m] = True
    idx = jnp.asarray(idx_np)
    w = jnp.asarray(w_np)
    # msk = jnp.asarray(msk_np)
    # return idx, w, msk  # shapes: (nz, Mmax), (nz, Mmax), (nz, Mmax)
    return idx, w  # shapes: (nz, Mmax), (nz, Mmax)


def get_loss_data_pdfs_ssfr_central(indir, nhalos):

    print("Loading PDF Mstar data...")

    fname = os.path.join(indir, "smdpl_smhm.h5")
    with h5py.File(fname, "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    fname = os.path.join(indir, "smdpl_smhm_samples_haloes.h5")
    with h5py.File(fname, "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    fname = os.path.join(indir, "smdpl_mstar_ssfr.h5")
    with h5py.File(fname, "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    mhalo_pdf_hist = hist / np.sum(hist, axis=1)[:, None]
    mhalo_pdf = counts / np.sum(counts, axis=1)[:, None]
    mhalo_pdf_cen = counts_cen / np.sum(counts_cen, axis=1)[:, None]
    mhalo_pdf_sat = counts_sat / np.sum(counts_sat, axis=1)[:, None]

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []

    mstar_ssfr_pdfs_cent = np.clip(mstar_ssfr_wcounts_cent, 0.0, None)
    mstar_ssfr_pdfs_cent = (
        mstar_ssfr_pdfs_cent
        / np.sum(mstar_ssfr_pdfs_cent, axis=(2, 3))[:, :, None, None]
    )
    mstar_ssfr_pdfs_cent = np.where(
        np.isnan(mstar_ssfr_pdfs_cent), 0.0, mstar_ssfr_pdfs_cent
    )
    mstar_ssfr_pdfs_cent = np.einsum(
        "zmab,zm->zab", mstar_ssfr_pdfs_cent, mhalo_pdf_cen
    )

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    index_mhalo = []
    indx_pdf = []
    _run_indx = 0
    for i in range(len(age_targets)):
        t_target = age_targets[i]
        index_mhalo_atz = []
        indx_pdf_atz = []
        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp == -1)

            if sel.sum() < 50:
                print(i, j)
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])

            index_mhalo_atz.append(j)
            indx_pdf_atz.append(_run_indx)
            _run_indx += 1

        index_mhalo.append(np.array(index_mhalo_atz))
        indx_pdf.append(np.array(indx_pdf_atz))
        # break
    # target_mstar_ids = np.array([4, 9, 14, 17, 19, 22])
    # target_mstar_ids = np.array([9, 14, 17, 19, 22])
    target_mstar_ids = np.array([8, 10, 13, 16, 19])
    # target_mstar_ids = np.array([9, 12, 15, 17, 19])
    # target_mstar_ids = np.array([14, 17, 19, 22])
    print(logmstar_binsc_pdf[target_mstar_ids])
    target_data = np.zeros(
        (len(age_targets), len(target_mstar_ids), len(logssfr_binsc_pdf))
    )
    for i in range(len(age_targets)):
        for j, jval in enumerate(target_mstar_ids):
            target_data[i, j] = (
                mstar_ssfr_pdfs_cent[i, jval] / mstar_ssfr_pdfs_cent[i, jval].sum()
            )

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)

    ran_key_data = jran.split(ran_key, len(t_obs_targets))

    ndbins_lo = []
    ndbins_hi = []
    for i in range(len(logmstar_bins_pdf) - 1):
        for j in range(len(logssfr_bins_pdf) - 1):
            ndbins_lo.append([logmstar_bins_pdf[i], logssfr_bins_pdf[j]])
            ndbins_hi.append([logmstar_bins_pdf[i + 1], logssfr_bins_pdf[j + 1]])
    ndbins_lo = np.array(ndbins_lo)
    ndbins_hi = np.array(ndbins_hi)

    indx_pdf, mhalo_pdf_cen_ragged = prepare_ragged(
        indx_pdf, mhalo_pdf_cen, index_mhalo
    )

    loss_data_ssfr = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins_pdf,
        logssfr_bins_pdf,
        mhalo_pdf_cen_ragged,
        indx_pdf,
        jnp.asarray(target_mstar_ids),
        target_data,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []

    mstar_ssfr_pdfs_cent = np.clip(mstar_ssfr_wcounts_cent, 0.0, None)
    mstar_ssfr_pdfs_cent = (
        mstar_ssfr_pdfs_cent
        / np.sum(mstar_ssfr_pdfs_cent, axis=(2, 3))[:, :, None, None]
    )
    mstar_ssfr_pdfs_cent = np.where(
        np.isnan(mstar_ssfr_pdfs_cent), 0.0, mstar_ssfr_pdfs_cent
    )
    mstar_ssfr_pdfs_cent = np.einsum(
        "zmab,zm->zab", mstar_ssfr_pdfs_cent, mhalo_pdf_cen
    )

    index_mhalo = []
    indx_pdf = []
    _run_indx = 0
    for i in range(len(age_targets)):
        t_target = age_targets[i]
        index_mhalo_atz = []
        indx_pdf_atz = []
        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp == -1)

            if sel.sum() < 50:
                print(i, j, t_target, logmh_binsc[j], sel.sum())
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])

            index_mhalo_atz.append(j)
            indx_pdf_atz.append(_run_indx)
            _run_indx += 1

        index_mhalo.append(np.array(index_mhalo_atz))
        indx_pdf.append(np.array(indx_pdf_atz))
        # break
    # target_mstar_ids = np.array([4, 9, 14, 17, 19, 22])
    # target_mstar_ids = np.array([9, 14, 17, 19, 22])
    target_mstar_ids = np.array([8, 10, 13, 16, 19])
    # target_mstar_ids = np.array([9, 12, 15, 17, 19])
    # target_mstar_ids = np.array([14, 17, 19, 22])
    print(logmstar_binsc_pdf[target_mstar_ids])
    target_data = np.zeros(
        (len(age_targets), len(target_mstar_ids), len(logssfr_binsc_pdf))
    )
    for i in range(len(age_targets)):
        for j, jval in enumerate(target_mstar_ids):
            target_data[i, j] = (
                mstar_ssfr_pdfs_cent[i, jval] / mstar_ssfr_pdfs_cent[i, jval].sum()
            )

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)

    ran_key_data = jran.split(ran_key, len(t_obs_targets))

    ndbins_lo = []
    ndbins_hi = []
    for i in range(len(logmstar_bins_pdf) - 1):
        for j in range(len(logssfr_bins_pdf) - 1):
            ndbins_lo.append([logmstar_bins_pdf[i], logssfr_bins_pdf[j]])
            ndbins_hi.append([logmstar_bins_pdf[i + 1], logssfr_bins_pdf[j + 1]])
    ndbins_lo = np.array(ndbins_lo)
    ndbins_hi = np.array(ndbins_hi)

    indx_pdf, mhalo_pdf_cen_ragged = prepare_ragged(
        indx_pdf, mhalo_pdf_cen, index_mhalo
    )

    loss_data_ssfr_pred = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins_pdf,
        logssfr_bins_pdf,
        mhalo_pdf_cen_ragged,
        indx_pdf,
        jnp.asarray(target_mstar_ids),
        target_data,
    )

    plot_data = (
        jnp.asarray(target_mstar_ids),
        logssfr_binsc_pdf,
        target_data,
        loss_data_ssfr_pred,
    )

    return loss_data_ssfr, plot_data


def get_loss_data_pdfs_ssfr_satellite(indir, nhalos):

    print("Loading PDF Mstar data...")

    fname = os.path.join(indir, "smdpl_smhm.h5")
    with h5py.File(fname, "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    fname = os.path.join(indir, "smdpl_smhm_samples_haloes.h5")
    with h5py.File(fname, "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    fname = os.path.join(indir, "smdpl_mstar_ssfr.h5")
    with h5py.File(fname, "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    mhalo_pdf_hist = hist / np.sum(hist, axis=1)[:, None]
    mhalo_pdf = counts / np.sum(counts, axis=1)[:, None]
    mhalo_pdf_cen = counts_cen / np.sum(counts_cen, axis=1)[:, None]
    mhalo_pdf_sat = counts_sat / np.sum(counts_sat, axis=1)[:, None]

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []

    mstar_ssfr_pdfs_sat = np.clip(mstar_ssfr_wcounts_sat, 0.0, None)
    mstar_ssfr_pdfs_sat = (
        mstar_ssfr_pdfs_sat / np.sum(mstar_ssfr_pdfs_sat, axis=(2, 3))[:, :, None, None]
    )
    mstar_ssfr_pdfs_sat = np.where(
        np.isnan(mstar_ssfr_pdfs_sat), 0.0, mstar_ssfr_pdfs_sat
    )
    mstar_ssfr_pdfs_sat = np.einsum("zmab,zm->zab", mstar_ssfr_pdfs_sat, mhalo_pdf_sat)

    index_mhalo = []
    indx_pdf = []
    _run_indx = 0

    tarr_logm0 = np.logspace(-1, LGT0, 50)
    for i in range(len(age_targets)):
        t_target = age_targets[i]
        index_mhalo_atz = []
        indx_pdf_atz = []
        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp != -1)

            if sel.sum() < 50:
                print(i, j)
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])

            index_mhalo_atz.append(j)
            indx_pdf_atz.append(_run_indx)
            _run_indx += 1

        index_mhalo.append(np.array(index_mhalo_atz))
        indx_pdf.append(np.array(indx_pdf_atz))

        # break
    # target_mstar_ids = np.array([4, 9, 14, 17, 19, 22])
    # target_mstar_ids = np.array([9, 14, 17, 19, 22])
    target_mstar_ids = np.array([8, 10, 13, 16, 19])
    # target_mstar_ids = np.array([9, 12, 15, 17])
    # target_mstar_ids = np.array([14, 17, 19, 22])
    print(logmstar_binsc_pdf[target_mstar_ids])
    target_data_sat = np.zeros(
        (len(age_targets), len(target_mstar_ids), len(logssfr_binsc_pdf))
    )
    for i in range(len(age_targets)):
        for j, jval in enumerate(target_mstar_ids):
            target_data_sat[i, j] = (
                mstar_ssfr_pdfs_sat[i, jval] / mstar_ssfr_pdfs_sat[i, jval].sum()
            )

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)

    ran_key_data = jran.split(ran_key, len(t_obs_targets))

    ndbins_lo = []
    ndbins_hi = []
    for i in range(len(logmstar_bins_pdf) - 1):
        for j in range(len(logssfr_bins_pdf) - 1):
            ndbins_lo.append([logmstar_bins_pdf[i], logssfr_bins_pdf[j]])
            ndbins_hi.append([logmstar_bins_pdf[i + 1], logssfr_bins_pdf[j + 1]])
    ndbins_lo = np.array(ndbins_lo)
    ndbins_hi = np.array(ndbins_hi)

    indx_pdf, mhalo_pdf_sat_ragged = prepare_ragged(
        indx_pdf, mhalo_pdf_sat, index_mhalo
    )

    loss_data_ssfr_sat = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins_pdf,
        logssfr_bins_pdf,
        mhalo_pdf_sat_ragged,
        indx_pdf,
        jnp.asarray(target_mstar_ids),
        target_data_sat,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []

    mstar_ssfr_pdfs_sat = np.clip(mstar_ssfr_wcounts_sat, 0.0, None)
    mstar_ssfr_pdfs_sat = (
        mstar_ssfr_pdfs_sat / np.sum(mstar_ssfr_pdfs_sat, axis=(2, 3))[:, :, None, None]
    )
    mstar_ssfr_pdfs_sat = np.where(
        np.isnan(mstar_ssfr_pdfs_sat), 0.0, mstar_ssfr_pdfs_sat
    )
    mstar_ssfr_pdfs_sat = np.einsum("zmab,zm->zab", mstar_ssfr_pdfs_sat, mhalo_pdf_sat)

    index_mhalo = []
    indx_pdf = []
    _run_indx = 0

    for i in range(len(age_targets)):
        t_target = age_targets[i]
        index_mhalo_atz = []
        indx_pdf_atz = []
        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp != -1)

            if sel.sum() < 50:
                print(i, j)
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])

            index_mhalo_atz.append(j)
            indx_pdf_atz.append(_run_indx)
            _run_indx += 1

        index_mhalo.append(np.array(index_mhalo_atz))
        indx_pdf.append(np.array(indx_pdf_atz))
        # break
    # target_mstar_ids = np.array([4, 9, 14, 17, 19, 22])
    # target_mstar_ids = np.array([9, 14, 17, 19, 22])
    target_mstar_ids = np.array([8, 10, 13, 16, 19])
    # target_mstar_ids = np.array([9, 12, 15, 17])
    # target_mstar_ids = np.array([14, 17, 19, 22])
    print(logmstar_binsc_pdf[target_mstar_ids])
    target_data_sat = np.zeros(
        (len(age_targets), len(target_mstar_ids), len(logssfr_binsc_pdf))
    )
    for i in range(len(age_targets)):
        for j, jval in enumerate(target_mstar_ids):
            target_data_sat[i, j] = (
                mstar_ssfr_pdfs_sat[i, jval] / mstar_ssfr_pdfs_sat[i, jval].sum()
            )

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)

    ran_key_data = jran.split(ran_key, len(t_obs_targets))

    ndbins_lo = []
    ndbins_hi = []
    for i in range(len(logmstar_bins_pdf) - 1):
        for j in range(len(logssfr_bins_pdf) - 1):
            ndbins_lo.append([logmstar_bins_pdf[i], logssfr_bins_pdf[j]])
            ndbins_hi.append([logmstar_bins_pdf[i + 1], logssfr_bins_pdf[j + 1]])
    ndbins_lo = np.array(ndbins_lo)
    ndbins_hi = np.array(ndbins_hi)

    indx_pdf, mhalo_pdf_sat_ragged = prepare_ragged(
        indx_pdf, mhalo_pdf_sat, index_mhalo
    )

    loss_data_ssfr_sat_pred = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins_pdf,
        logssfr_bins_pdf,
        mhalo_pdf_sat_ragged,
        indx_pdf,
        jnp.asarray(target_mstar_ids),
        target_data_sat,
    )

    plot_data = (
        jnp.asarray(target_mstar_ids),
        logssfr_binsc_pdf,
        target_data_sat,
        loss_data_ssfr_sat_pred,
    )

    return loss_data_ssfr_sat, plot_data


def get_loss_data_pdfs_mstar_cen(indir, nhalos):
    # Load PDF data ---------------------------------------------
    print("Loading PDF Mstar for centrals data...")

    fname = os.path.join(indir, "smdpl_smhm.h5")
    with h5py.File(fname, "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    fname = os.path.join(indir, "smdpl_smhm_samples_haloes.h5")
    with h5py.File(fname, "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    fname = os.path.join(indir, "smdpl_mstar_ssfr.h5")
    with h5py.File(fname, "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    mstar_wcounts_cen = np.sum(mstar_ssfr_wcounts_cent, axis=3)
    mstar_wcounts_sat = np.sum(mstar_ssfr_wcounts_sat, axis=3)

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    # Centrals
    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp == -1)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(
                mstar_wcounts_cen[i, j] / mstar_wcounts_cen[i, j].sum()
            )
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000
    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp == -1)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(
                mstar_wcounts_cen[i, j] / mstar_wcounts_cen[i, j].sum()
            )
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar_pred = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )
    plot_data = (
        logmstar_bins_pdf,
        mstar_wcounts,
        age_targets,
        redshift_targets,
        tobs_id,
        logmh_id,
        logmh_binsc,
        loss_data_mstar_pred,
    )

    return loss_data_mstar, plot_data


def get_loss_data_pdfs_mstar_sat(indir, nhalos):
    # Load PDF data ---------------------------------------------
    print("Loading PDF Mstar for satellites data...")

    fname = os.path.join(indir, "smdpl_smhm.h5")
    with h5py.File(fname, "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    fname = os.path.join(indir, "smdpl_smhm_samples_haloes.h5")
    with h5py.File(fname, "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    fname = os.path.join(indir, "smdpl_mstar_ssfr.h5")
    with h5py.File(fname, "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    mstar_wcounts_cen = np.sum(mstar_ssfr_wcounts_cent, axis=3)
    mstar_wcounts_sat = np.sum(mstar_ssfr_wcounts_sat, axis=3)

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    # Centrals
    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp != -1)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(
                mstar_wcounts_sat[i, j] / mstar_wcounts_sat[i, j].sum()
            )
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000
    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = -99.0  # 2.0

    mah_params_data = []
    logmp0_data = []
    upid_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j) & (upid_samp != -1)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            upid_data.append(upid_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(
                mstar_wcounts_sat[i, j] / mstar_wcounts_sat[i, j].sum()
            )
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            logmp0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    logmp0_data = np.array(logmp0_data)
    upid_data = np.array(upid_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar_pred = (
        mah_params_data,
        logmp0_data,
        upid_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )
    plot_data = (
        logmstar_bins_pdf,
        mstar_wcounts,
        age_targets,
        redshift_targets,
        tobs_id,
        logmh_id,
        logmh_binsc,
        loss_data_mstar_pred,
    )

    return loss_data_mstar, plot_data
