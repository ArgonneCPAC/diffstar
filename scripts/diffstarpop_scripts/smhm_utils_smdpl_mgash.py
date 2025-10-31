""" """

import os
import re
import h5py
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop
from diffsky.diffndhist import tw_ndhist_weighted
from diffstar.defaults_mgash_model import DEFAULT_DIFFSTAR_PARAMS, LGT0, T_TABLE_MIN
from diffstar.sfh_model_mgash import calc_sfh_galpop
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13
from umachine_pyio.load_mock import load_mock_from_binaries

LCRC_NOMERGING_DIFFSTAR_DRN = (
    "/lcrc/project/halotools/alarcon/results/mgash/UniverseMachine/DR1_nomerging/"
)
LCRC_NOMERGING_DIFFMAH_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffmah_tpeak_fits/"
)
LCRC_NOMERGING_BINARIES_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/"
)

LCRC_DR1_DIFFSTAR_DRN = (
    "/lcrc/project/halotools/alarcon/results/mgash/UniverseMachine/DR1/"
)
LCRC_DR1_DIFFMAH_DRN = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/diffmah_tpeak_fits/"
LCRC_DR1_BINARIES_DRN = (
    "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/a_1.000000/"
)
LCRC_NOMERGING_diffstar_bnpat = "diffstar_fits_subvol_{}.hdf5"
LCRC_DR1_diffstar_bnpat = "diffstar_fits_subvol_{}.hdf5"
LCRC_NOMERGING_diffmah_bnpat = "subvol_{}_diffmah_fits.h5"

TASSO_DIFFSTAR_DRN = "/Users/aphearin/work/DATA/diffstar_data/SMDPL/"
N_SUBVOL_SMDPL = 576

LGMH_MIN, LGMH_MAX = 11, 14.75
N_LGM_BINS = 12
LOGMH_BINS = np.linspace(LGMH_MIN, LGMH_MAX, N_LGM_BINS)

LOGMSTAR_BINS_PDF = np.linspace(7.0, 13.0, 26)
LOGSSFR_BINS_PDF = np.linspace(-13.0, -8.0, 30)

Z_BINS = [0.0, 0.5, 1.0, 1.5, 2.0]

T0_SMDPL = 13.7976158

N_HALOS_MAX = 20_000
N_HALOS_PER_SUBVOL = N_HALOS_MAX // N_SUBVOL_SMDPL


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data


def return_subvol_str_diffmah(subvol, sim_name, diffstar_drn, diffstar_bnpat):
    regex_str = re.escape(diffstar_bnpat).replace(r"\{\}", r"(\d{1,3})")
    pattern = re.compile(f"^{regex_str}$")
    matching_files = [f for f in os.listdir(diffstar_drn) if pattern.match(f)]
    if sim_name == "DR1_nomerging":
        subvols = [x.split("_")[1] for x in matching_files]
    elif sim_name == "DR1":
        subvols = [x.split("_")[-1].split(".")[0] for x in matching_files]
    subvols_len = np.array([len(x) for x in subvols])

    if np.any(subvols_len == 1):
        subvol_str = f"{subvol:d}"
    elif np.all(subvols_len == subvols_len.max()):
        nchar_subvol = subvols_len.max()
        subvol_str = f"{subvol:0{nchar_subvol}d}"
    return subvol_str


def return_subvol_str(subvol, sim_name, diffstar_drn, diffstar_bnpat):
    regex_str = re.escape(diffstar_bnpat).replace(r"\{\}", r"(\d{1,3})")
    pattern = re.compile(f"^{regex_str}$")
    matching_files = [f for f in os.listdir(diffstar_drn) if pattern.match(f)]
    subvols = [x.split("_")[-1].split(".")[0] for x in matching_files]
    subvols_len = np.array([len(x) for x in subvols])

    if np.any(subvols_len == 1):
        subvol_str = f"{subvol:d}"
    elif np.all(subvols_len == subvols_len.max()):
        nchar_subvol = subvols_len.max()
        subvol_str = f"{subvol:0{nchar_subvol}d}"
    return subvol_str


def load_diffstar_subvolume(
    subvol,
    sim_name,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
    diffstar_bnpat=LCRC_NOMERGING_diffstar_bnpat,
):
    # nchar_subvol = len(str(n_subvol_tot))
    subvol_str = return_subvol_str(subvol, sim_name, diffstar_drn, diffstar_bnpat)

    diffstar_bn = diffstar_bnpat.format(subvol_str)
    diffstar_fn = os.path.join(diffstar_drn, diffstar_bn)
    diffstar_data = _load_flat_hdf5(diffstar_fn)

    if sim_name == "DR1_nomerging":
        subvol_str = return_subvol_str_diffmah(
            subvol, sim_name, diffmah_drn, LCRC_NOMERGING_diffmah_bnpat
        )
        diffmah_bn = LCRC_NOMERGING_diffmah_bnpat.format(subvol_str).replace(
            "diffstar", "diffmah"
        )
    elif sim_name == "DR1":
        diffmah_bn = diffstar_bn.replace("diffstar", "diffmah")
    diffmah_fn = os.path.join(diffmah_drn, diffmah_bn)
    diffmah_data = _load_flat_hdf5(diffmah_fn)

    return diffmah_data, diffstar_data


def load_diffstar_sfh_tables(
    subvol,
    sim_name,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
    diffstar_bnpat=LCRC_NOMERGING_diffstar_bnpat,
    lgt0=LGT0,
    n_times=200,
):
    diffmah_data, diffstar_data = load_diffstar_subvolume(
        subvol,
        sim_name,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
        diffstar_bnpat=diffstar_bnpat,
    )
    has_fit = (diffmah_data["loss"] > 0.0) & (diffstar_data["success"] == 1)
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key][has_fit] for key in DEFAULT_MAH_PARAMS._fields]
    )

    ms_params = DEFAULT_DIFFSTAR_PARAMS.ms_params._make(
        [
            diffstar_data[key][has_fit]
            for key in DEFAULT_DIFFSTAR_PARAMS.ms_params._fields
        ]
    )
    q_params = DEFAULT_DIFFSTAR_PARAMS.q_params._make(
        [
            diffstar_data[key][has_fit]
            for key in DEFAULT_DIFFSTAR_PARAMS.q_params._fields
        ]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make((ms_params, q_params))

    t_0 = 10**lgt0
    t_table = np.linspace(T_TABLE_MIN, t_0, n_times)

    __, log_mah_table = mah_halopop(mah_params, t_table, LGT0)

    sfh_table, smh_table = calc_sfh_galpop(
        sfh_params, mah_params, t_table, lgt0=LGT0, return_smh=True
    )
    log_sfh_table = np.log10(sfh_table)
    log_smh_table = np.log10(smh_table)
    log_ssfrh_table = log_sfh_table - log_smh_table

    out = (
        t_table,
        log_mah_table,
        log_smh_table,
        log_ssfrh_table,
        mah_params,
        ms_params,
        q_params,
        has_fit,
    )

    return out


def compute_weighted_histograms_z0(
    subvol,
    sim_name,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
    diffstar_bnpat=LCRC_NOMERGING_diffstar_bnpat,
    lgt0=LGT0,
    logmh_bins=LOGMH_BINS,
):
    _res = load_diffstar_sfh_tables(
        subvol,
        sim_name,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
        diffstar_bnpat=diffstar_bnpat,
        lgt0=lgt0,
    )
    t_table, log_mah_table, log_smh_table, log_ssfrh_table = _res[:4]

    n_halos = log_smh_table.shape[0]

    nddata = log_mah_table[:, -1].reshape((-1, 1))

    sigma = np.mean(np.diff(logmh_bins)) + np.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table[:, -1].reshape((-1, 1))
    _ones = np.ones_like(ydata)

    ndbins_lo = logmh_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmh_bins[1:].reshape((-1, 1))

    whist = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    return wcounts, whist


def compute_diff_histograms_atz(logmh_bins, log_mah_table, log_smh_table):

    n_halos = log_smh_table.shape[0]

    nddata = log_mah_table.reshape((-1, 1))

    sigma = np.mean(np.diff(logmh_bins)) + np.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table.reshape((-1, 1))
    _ones = np.ones_like(ydata)

    ndbins_lo = logmh_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmh_bins[1:].reshape((-1, 1))

    whist = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    return wcounts, whist


def compute_histograms_atz(logmh_bins, log_mah_table, log_smh_table):
    count = binned_statistic(
        log_mah_table, values=log_smh_table, bins=logmh_bins, statistic="count"
    )[0]
    whist = binned_statistic(
        log_mah_table, values=log_smh_table, bins=logmh_bins, statistic="sum"
    )[0]
    return count, whist


def get_redshift_from_age(age):
    z_table = np.linspace(0, 10, 2000)[::-1]
    t_table = Planck13.age(z_table).value
    redshift_from_age = np.interp(age, t_table, z_table)
    return redshift_from_age


def return_target_redshfit_index(t_table, redshift_targets):
    z_table = get_redshift_from_age(t_table)
    return np.digitize(redshift_targets, z_table)


def sample_halos(
    n_subvol_smdpl,
    logmh_bins,
    log_mah,
    log_smh,
    mah_params,
    ms_params,
    q_params,
    upid,
):
    ndbins_lo = logmh_bins[:-1]
    ndbins_hi = logmh_bins[1:]
    arange_arr = np.arange(len(log_mah))
    logmh_id = []
    logmh_val = []
    mah_params_samp = []
    ms_params_samp = []
    q_params_samp = []
    upid_samp = []

    mah_params = np.array(mah_params).T
    ms_params = np.array(ms_params).T
    q_params = np.array(q_params).T

    n_halos_per_subvol = N_HALOS_MAX // n_subvol_smdpl

    for i in range(len(ndbins_lo)):
        sel = (log_mah >= ndbins_lo[i]) & (log_mah < ndbins_hi[i])
        sel_num = int(min(n_halos_per_subvol, sel.sum()))
        sel = np.random.choice(arange_arr[sel], sel_num, replace=False)
        logmh_id.append(np.ones_like(sel) * i)
        logmh_val.append(np.ones_like(sel) * ((ndbins_lo[i] + ndbins_hi[i]) / 2.0))
        mah_params_samp.append(mah_params[sel])
        ms_params_samp.append(ms_params[sel])
        q_params_samp.append(q_params[sel])
        upid_samp.append(upid[sel])

    logmh_id = np.concatenate(logmh_id)
    logmh_val = np.concatenate(logmh_val)
    mah_params_samp = np.concatenate(mah_params_samp)
    ms_params_samp = np.concatenate(ms_params_samp)
    q_params_samp = np.concatenate(q_params_samp)
    upid_samp = np.concatenate(upid_samp)

    mah_params_samp = DEFAULT_MAH_PARAMS._make(mah_params_samp.T)
    ms_params_samp = DEFAULT_DIFFSTAR_PARAMS.ms_params._make(ms_params_samp.T)
    q_params_samp = DEFAULT_DIFFSTAR_PARAMS.q_params._make(q_params_samp.T)
    out = (
        logmh_id,
        logmh_val,
        mah_params_samp,
        ms_params_samp,
        q_params_samp,
        upid_samp,
    )
    return out


def create_target_data(
    subvol,
    sim_name,
    n_subvol_smdpl,
    redshift_targets=Z_BINS,
    n_subvol_tot=N_SUBVOL_SMDPL,
    binaries_drn=LCRC_NOMERGING_BINARIES_DRN,
    diffmah_drn=LCRC_NOMERGING_DIFFMAH_DRN,
    diffstar_drn=LCRC_NOMERGING_DIFFSTAR_DRN,
    diffstar_bnpat=LCRC_NOMERGING_diffstar_bnpat,
    lgt0=LGT0,
    logmh_bins=LOGMH_BINS,
):
    _res = load_diffstar_sfh_tables(
        subvol,
        sim_name,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
        diffstar_bnpat=diffstar_bnpat,
        lgt0=lgt0,
    )
    (
        t_table,
        log_mah_table,
        log_smh_table,
        log_ssfrh_table,
        mah_params,
        ms_params,
        q_params,
        has_fit,
    ) = _res

    galprops = ["halo_id", "upid"]
    halos = load_mock_from_binaries(
        np.atleast_1d(subvol), root_dirname=binaries_drn, galprops=galprops
    )
    upid = np.array(halos["upid"])[has_fit]

    tids = return_target_redshfit_index(t_table, redshift_targets)

    nz, nm = len(redshift_targets), len(logmh_bins) - 1

    wcounts_zid = np.zeros((nz, nm))
    whist_zid = np.zeros((nz, nm))
    counts_zid = np.zeros((nz, nm))
    hist_zid = np.zeros((nz, nm))

    counts_zid_cen = np.zeros((nz, nm))
    counts_zid_sat = np.zeros((nz, nm))

    for i, tid in enumerate(tids):
        _res = compute_diff_histograms_atz(
            logmh_bins, log_mah_table[:, tid], log_smh_table[:, tid]
        )
        wcounts_zid[i] = _res[0]
        whist_zid[i] = _res[1]

        _res = compute_histograms_atz(
            logmh_bins, log_mah_table[:, tid], log_smh_table[:, tid]
        )
        counts_zid[i] = _res[0]
        hist_zid[i] = _res[1]

        is_central = upid == -1
        counts_zid_cen[i] = compute_histograms_atz(
            logmh_bins,
            log_mah_table[:, tid][is_central],
            log_smh_table[:, tid][is_central],
        )[0]
        counts_zid_sat[i] = compute_histograms_atz(
            logmh_bins,
            log_mah_table[:, tid][~is_central],
            log_smh_table[:, tid][~is_central],
        )[0]

    data = []

    for i, tid in enumerate(tids):
        _res = sample_halos(
            n_subvol_smdpl,
            logmh_bins,
            log_mah_table[:, tid],
            log_smh_table[:, tid],
            mah_params,
            ms_params,
            q_params,
            upid,
        )
        data.append(
            (
                *_res,
                np.ones_like(_res[0]) * i,
                np.ones_like(_res[0]) * t_table[tid],
                np.ones_like(_res[0]) * redshift_targets[i],
            )
        )

    haloes = concatenate_samples_haloes(data)

    out = (
        wcounts_zid,
        whist_zid,
        counts_zid,
        hist_zid,
        t_table[tids],
        haloes,
        counts_zid_cen,
        counts_zid_sat,
    )

    return out


def concatenate_samples_haloes(data):
    logmh_id = []
    logmh_val = []
    redshift_val = []
    tobs_id = []
    tobs_val = []
    mah_params_samp = []
    ms_params_samp = []
    q_params_samp = []
    upid_samp = []

    for subdata in data:

        logmh_id.append(subdata[0])
        logmh_val.append(subdata[1])
        mah_params_samp.append(np.array(subdata[2]).T)
        ms_params_samp.append(np.array(subdata[3]).T)
        q_params_samp.append(np.array(subdata[4]).T)
        upid_samp.append(subdata[5])
        tobs_id.append(subdata[6])
        tobs_val.append(subdata[7])
        redshift_val.append(subdata[8])

    logmh_id = np.concatenate(logmh_id)
    logmh_val = np.concatenate(logmh_val)
    mah_params_samp = np.concatenate(mah_params_samp)
    ms_params_samp = np.concatenate(ms_params_samp)
    q_params_samp = np.concatenate(q_params_samp)
    upid_samp = np.concatenate(upid_samp)
    tobs_id = np.concatenate(tobs_id)
    tobs_val = np.concatenate(tobs_val)
    redshift_val = np.concatenate(redshift_val)

    mah_params_samp = DEFAULT_MAH_PARAMS._make(mah_params_samp.T)
    ms_params_samp = DEFAULT_DIFFSTAR_PARAMS.ms_params._make(ms_params_samp.T)
    q_params_samp = DEFAULT_DIFFSTAR_PARAMS.q_params._make(q_params_samp.T)

    haloes = (
        logmh_id,
        logmh_val,
        mah_params_samp,
        ms_params_samp,
        q_params_samp,
        upid_samp,
        tobs_id,
        tobs_val,
        redshift_val,
    )
    return haloes


def compute_diff_histograms_mstar_atmobs_z(
    logmstar_bins,
    log_smh_table,
):

    n_halos = log_smh_table.shape[0]

    nddata = log_smh_table.reshape((-1, 1))

    sigma = np.mean(np.diff(logmstar_bins)) + np.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table.reshape((-1, 1))
    _ones = np.ones_like(ydata)

    ndbins_lo = logmstar_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmstar_bins[1:].reshape((-1, 1))

    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    counts = np.histogram(ydata, logmstar_bins)[0]

    return wcounts, counts


def compute_diff_histograms_mstar_ssfr_atz(
    log_smh_table,
    log_ssfr_table,
    ndbins_lo,
    ndbins_hi,
    logmstar_bins,
    logssfr_bins,
):
    n_halos = log_smh_table.shape[0]

    sigma_mstar = np.mean(np.diff(logmstar_bins)) + np.zeros(n_halos)
    sigma_ssfr = np.mean(np.diff(logssfr_bins)) + np.zeros(n_halos)

    ndsig = np.ones((n_halos, 2))
    ndsig[:, 0] = sigma_mstar
    ndsig[:, 1] = sigma_ssfr

    nddata = np.array([log_smh_table, log_ssfr_table]).T

    _ones = np.ones(n_halos)

    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    return wcounts


def create_pdf_target_data(
    subvol,
    sim_name,
    redshift_targets=Z_BINS,
    n_subvol_tot=N_SUBVOL_SMDPL,
    binaries_drn=LCRC_NOMERGING_BINARIES_DRN,
    diffmah_drn=LCRC_NOMERGING_DIFFMAH_DRN,
    diffstar_drn=LCRC_NOMERGING_DIFFSTAR_DRN,
    diffstar_bnpat=LCRC_NOMERGING_diffstar_bnpat,
    lgt0=LGT0,
    logmh_bins=LOGMH_BINS,
    logmstar_bins_pdf=LOGMSTAR_BINS_PDF,
    logssfr_bins_pdf=LOGSSFR_BINS_PDF,
):
    _res = load_diffstar_sfh_tables(
        subvol,
        sim_name,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
        diffstar_bnpat=diffstar_bnpat,
        lgt0=lgt0,
    )
    (
        t_table,
        log_mah_table,
        log_smh_table,
        log_ssfrh_table,
        mah_params,
        ms_params,
        q_params,
        has_fit,
    ) = _res

    log_ssfrh_table = np.clip(log_ssfrh_table, -12.0, None)

    galprops = ["halo_id", "upid"]
    halos = load_mock_from_binaries(
        np.atleast_1d(subvol), root_dirname=binaries_drn, galprops=galprops
    )
    upid = np.array(halos["upid"])[has_fit]
    is_central = upid == -1

    tids = return_target_redshfit_index(t_table, redshift_targets)

    nz, nm = len(redshift_targets), len(logmh_bins) - 1
    nmstar = len(logmstar_bins_pdf) - 1
    nssfr = len(logssfr_bins_pdf) - 1

    ndbins_lo = []
    ndbins_hi = []
    for i in range(len(logmstar_bins_pdf) - 1):
        for j in range(len(logssfr_bins_pdf) - 1):
            ndbins_lo.append([logmstar_bins_pdf[i], logssfr_bins_pdf[j]])
            ndbins_hi.append([logmstar_bins_pdf[i + 1], logssfr_bins_pdf[j + 1]])
    ndbins_lo = np.array(ndbins_lo)
    ndbins_hi = np.array(ndbins_hi)

    mstar_wcounts = np.zeros((nz, nm, nmstar))
    mstar_counts = np.zeros((nz, nm, nmstar))

    mstar_ssfr_wcounts_cent = np.zeros((nz, nm, nmstar, nssfr))
    mstar_ssfr_wcounts_sat = np.zeros((nz, nm, nmstar, nssfr))

    for i, tid in enumerate(tids):
        for j in range(nm):
            mobs_sel = (log_mah_table[:, tid] > logmh_bins[j]) & (
                log_mah_table[:, tid] < logmh_bins[j + 1]
            )
            _res = compute_diff_histograms_mstar_atmobs_z(
                logmstar_bins_pdf,
                log_smh_table[mobs_sel][:, tid],
            )
            mstar_wcounts[i, j] = _res[0]
            mstar_counts[i, j] = _res[1]

            mobs_sel_cent = mobs_sel & is_central
            _res = compute_diff_histograms_mstar_ssfr_atz(
                log_smh_table[mobs_sel_cent][:, tid],
                log_ssfrh_table[mobs_sel_cent][:, tid],
                ndbins_lo,
                ndbins_hi,
                logmstar_bins_pdf,
                logssfr_bins_pdf,
            )
            mstar_ssfr_wcounts_cent[i, j] = _res.reshape((nmstar, nssfr))

            mobs_sel_sat = mobs_sel & (~is_central)
            _res = compute_diff_histograms_mstar_ssfr_atz(
                log_smh_table[mobs_sel_sat][:, tid],
                log_ssfrh_table[mobs_sel_sat][:, tid],
                ndbins_lo,
                ndbins_hi,
                logmstar_bins_pdf,
                logssfr_bins_pdf,
            )
            mstar_ssfr_wcounts_sat[i, j] = _res.reshape((nmstar, nssfr))

    out = (
        mstar_wcounts,
        mstar_counts,
        mstar_ssfr_wcounts_cent,
        mstar_ssfr_wcounts_sat,
    )

    return out
