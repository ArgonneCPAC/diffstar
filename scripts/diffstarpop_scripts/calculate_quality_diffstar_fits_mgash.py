import re
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
import h5py

import argparse

from astropy.cosmology import Planck15, z_at_value

mred = "#d62728"
morange = "#ff7f0e"
mgreen = "#2ca02c"
mblue = "#1f77b4"
mpurple = "#9467bd"
plt.rc("font", family="serif")
plt.rc("font", size=22)
plt.rc("text", usetex=False)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")  # necessary to use \dfrac

import smhm_utils_tng_mgash
import smhm_utils_galacticus_mgash
import smhm_utils_smdpl_mgash
from smhm_utils_smdpl_mgash import load_diffstar_sfh_tables

from diffstar.data_loaders.load_smah_data import (
    FB_SMDPL,
    T0_SMDPL,
    load_smdpl_diffmah_fits,
    load_SMDPL_DR1_data,
    load_SMDPL_nomerging_data,
    load_tng_data,
)
from diffstar.data_loaders.load_galacticus_sfh import load_galacticus_diffstar_data

from jax import vmap, jit as jjit, numpy as jnp
from diffstar.defaults import TODAY, LGT0
from diffstar.utils import cumulative_mstar_formed_galpop
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop, DEFAULT_MAH_PARAMS


def _jnp_interp_vmap(x_new, x, y):
    return jnp.interp(x_new, x, y)


jnp_interp_vmap = jjit(vmap(_jnp_interp_vmap, in_axes=(None, None, 0)))


def calculate_plot_smdpl_nomerging(mpeak_bins):
    diffmah_drn = smhm_utils_smdpl_mgash.LCRC_NOMERGING_DIFFMAH_DRN
    diffstar_drn = smhm_utils_smdpl_mgash.LCRC_NOMERGING_DIFFSTAR_DRN
    binaries_drn = smhm_utils_smdpl_mgash.LCRC_NOMERGING_BINARIES_DRN
    diffstar_bnpat = smhm_utils_smdpl_mgash.LCRC_NOMERGING_diffstar_bnpat
    sim_name = "DR1_nomerging"

    regex_str = re.escape(diffstar_bnpat).replace(r"\{\}", r"(\d{1,3})")
    pattern = re.compile(f"^{regex_str}$")
    matching_files = [f for f in os.listdir(diffstar_drn) if pattern.match(f)]
    subvols = [x.split("_")[-1].split(".")[0] for x in matching_files]
    subvols = np.sort(np.array(subvols).astype(int))
    n_subvol_smdpl = len(subvols)

    mpeak_binsc = 0.5 * (mpeak_bins[1:] + mpeak_bins[:-1])
    nt = 117

    mstar_data_mean = np.zeros((len(mpeak_binsc), nt))
    mstar_fit_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_data_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_fit_mean = np.zeros((len(mpeak_binsc), nt))

    ngals = np.zeros(len(mpeak_binsc))

    for subvol in range(576):

        print(subvol)

        out = smhm_utils_smdpl_mgash.load_diffstar_sfh_tables(
            subvol,
            sim_name,
            n_subvol_smdpl,
            diffmah_drn,
            diffstar_drn,
            diffstar_bnpat,
        )
        (
            t_table,
            log_mah_table,
            log_smh_table,
            log_ssfrh_table,
            mah_params,
            sfh_params,
            has_fit,
        ) = out

        log_sfh_table = log_ssfrh_table + log_smh_table

        out = load_SMDPL_nomerging_data([subvol], binaries_drn)
        (halo_ids, log_smahs, sfrh, SMDPL_t, log_mahs, logmp0) = out
        log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)

        _log_smahs_data = log_smahs[has_fit]
        _log_sfrh_data = log_sfrh[has_fit]

        _log_smahs_fits = jnp_interp_vmap(SMDPL_t, t_table, log_smh_table)
        _log_sfrh_fits = jnp_interp_vmap(SMDPL_t, t_table, log_sfh_table)

        smahs_fits = np.where(_log_smahs_fits == 0.0, np.nan, 10**_log_smahs_fits)
        sfrh_fits = np.where(_log_sfrh_fits == 0.0, np.nan, 10**_log_sfrh_fits)
        smahs_data = np.where(_log_smahs_data == 0.0, np.nan, 10**_log_smahs_data)
        sfrh_data = np.where(_log_sfrh_data == 0.0, np.nan, 10**_log_sfrh_data)

        logmp0_data = logmp0[has_fit]

        ssfrh = sfrh_data / smahs_data
        ssfrh_fit = sfrh_fits / smahs_fits
        ssfrh = np.clip(ssfrh, 1e-12, np.inf)
        ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
        sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
        sfrh_fits = ssfrh_fit * smahs_fits

        for i in range(len(mpeak_bins) - 1):
            masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mstar_data_mean[i] += np.nansum(smahs_data[masksel], axis=0)
                mstar_fit_mean[i] += np.nansum(smahs_fits[masksel], axis=0)
                sfr_data_mean[i] += np.nansum(sfrh[masksel], axis=0)
                sfr_fit_mean[i] += np.nansum(sfrh_fits[masksel], axis=0)

                ngals[i] += masksel.sum()

    mstar_data_mean /= ngals[:, None]
    mstar_fit_mean /= ngals[:, None]
    sfr_data_mean /= ngals[:, None]
    sfr_fit_mean /= ngals[:, None]

    out = (
        mpeak_bins,
        mpeak_binsc,
        SMDPL_t,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    )

    return out


def calculate_plot_smdpl_dr1(mpeak_bins):
    diffmah_drn = smhm_utils_smdpl_mgash.LCRC_DR1_DIFFMAH_DRN
    diffstar_drn = smhm_utils_smdpl_mgash.LCRC_DR1_DIFFSTAR_DRN
    binaries_drn = smhm_utils_smdpl_mgash.LCRC_DR1_BINARIES_DRN
    diffstar_bnpat = smhm_utils_smdpl_mgash.LCRC_DR1_diffstar_bnpat
    sim_name = "DR1"

    regex_str = re.escape(diffstar_bnpat).replace(r"\{\}", r"(\d{1,3})")
    pattern = re.compile(f"^{regex_str}$")
    matching_files = [f for f in os.listdir(diffstar_drn) if pattern.match(f)]
    subvols = [x.split("_")[-1].split(".")[0] for x in matching_files]
    subvols = np.sort(np.array(subvols).astype(int))
    n_subvol_smdpl = len(subvols)

    mpeak_binsc = 0.5 * (mpeak_bins[1:] + mpeak_bins[:-1])
    nt = 117

    mstar_data_mean = np.zeros((len(mpeak_binsc), nt))
    mstar_fit_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_data_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_fit_mean = np.zeros((len(mpeak_binsc), nt))

    ngals = np.zeros(len(mpeak_binsc))

    print(n_subvol_smdpl)

    for subvol in subvols:

        print(subvol)

        out = smhm_utils_smdpl_mgash.load_diffstar_sfh_tables(
            subvol,
            sim_name,
            n_subvol_smdpl,
            diffmah_drn,
            diffstar_drn,
            diffstar_bnpat,
        )
        (
            t_table,
            log_mah_table,
            log_smh_table,
            log_ssfrh_table,
            mah_params,
            sfh_params,
            has_fit,
        ) = out

        log_sfh_table = log_ssfrh_table + log_smh_table

        out = load_SMDPL_DR1_data([subvol], binaries_drn)
        (halo_ids, log_smahs, sfrh, SMDPL_t, log_mahs, logmp0) = out
        log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)

        _log_smahs_data = log_smahs[has_fit]
        _log_sfrh_data = log_sfrh[has_fit]

        _log_smahs_fits = jnp_interp_vmap(SMDPL_t, t_table, log_smh_table)
        _log_sfrh_fits = jnp_interp_vmap(SMDPL_t, t_table, log_sfh_table)

        smahs_fits = np.where(_log_smahs_fits == 0.0, np.nan, 10**_log_smahs_fits)
        sfrh_fits = np.where(_log_sfrh_fits == 0.0, np.nan, 10**_log_sfrh_fits)
        smahs_data = np.where(_log_smahs_data == 0.0, np.nan, 10**_log_smahs_data)
        sfrh_data = np.where(_log_sfrh_data == 0.0, np.nan, 10**_log_sfrh_data)

        logmp0_data = logmp0[has_fit]

        ssfrh = sfrh_data / smahs_data
        ssfrh_fit = sfrh_fits / smahs_fits
        ssfrh = np.clip(ssfrh, 1e-12, np.inf)
        ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
        sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
        sfrh_fits = ssfrh_fit * smahs_fits

        for i in range(len(mpeak_bins) - 1):
            masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mstar_data_mean[i] += np.nansum(smahs_data[masksel], axis=0)
                mstar_fit_mean[i] += np.nansum(smahs_fits[masksel], axis=0)
                sfr_data_mean[i] += np.nansum(sfrh[masksel], axis=0)
                sfr_fit_mean[i] += np.nansum(sfrh_fits[masksel], axis=0)

                ngals[i] += masksel.sum()

    mstar_data_mean /= ngals[:, None]
    mstar_fit_mean /= ngals[:, None]
    sfr_data_mean /= ngals[:, None]
    sfr_fit_mean /= ngals[:, None]

    out = (
        mpeak_bins,
        mpeak_binsc,
        SMDPL_t,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    )

    return out


def calculate_plot_tng(mpeak_bins):
    diffmah_drn = smhm_utils_tng_mgash.BEBOP_TNG_MAH
    diffstar_drn = smhm_utils_tng_mgash.BEBOP_TNG_SFH
    binaries_drn = smhm_utils_tng_mgash.BEBOP_TNG

    mpeak_binsc = 0.5 * (mpeak_bins[1:] + mpeak_bins[:-1])
    out = load_tng_data(binaries_drn)
    (halo_ids, log_smahs, sfrh, tng_t, log_mahs, logmp0) = out
    log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)
    nt = len(tng_t)
    n_subvol_smdpl = 20

    nhalos_tot = len(halo_ids)

    _a = np.arange(0, nhalos_tot).astype("i8")

    mstar_data_mean = np.zeros((len(mpeak_binsc), nt))
    mstar_fit_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_data_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_fit_mean = np.zeros((len(mpeak_binsc), nt))

    ngals = np.zeros(len(mpeak_binsc))

    print(n_subvol_smdpl)

    for subvol in range(20):

        print(subvol)

        indx = np.array_split(_a, n_subvol_smdpl)[subvol]

        out = smhm_utils_tng_mgash.load_diffstar_sfh_tables(
            subvol,
            diffmah_drn,
            diffstar_drn,
        )
        (
            t_table,
            log_mah_table,
            log_smh_table,
            log_ssfrh_table,
            mah_params,
            sfh_params,
            has_fit,
        ) = out

        log_sfh_table = log_ssfrh_table + log_smh_table

        _log_smahs_data = log_smahs[indx][has_fit]
        _log_sfrh_data = log_sfrh[indx][has_fit]

        _log_smahs_fits = jnp_interp_vmap(tng_t, t_table, log_smh_table)
        _log_sfrh_fits = jnp_interp_vmap(tng_t, t_table, log_sfh_table)

        smahs_fits = np.where(_log_smahs_fits == 0.0, np.nan, 10**_log_smahs_fits)
        sfrh_fits = np.where(_log_sfrh_fits == 0.0, np.nan, 10**_log_sfrh_fits)
        smahs_data = np.where(_log_smahs_data == 0.0, np.nan, 10**_log_smahs_data)
        sfrh_data = np.where(_log_sfrh_data == 0.0, np.nan, 10**_log_sfrh_data)

        logmp0_data = logmp0[indx][has_fit]

        ssfrh = sfrh_data / smahs_data
        ssfrh_fit = sfrh_fits / smahs_fits
        ssfrh = np.clip(ssfrh, 1e-12, np.inf)
        ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
        sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
        sfrh_fits = ssfrh_fit * smahs_fits

        for i in range(len(mpeak_bins) - 1):
            masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mstar_data_mean[i] += np.nansum(smahs_data[masksel], axis=0)
                mstar_fit_mean[i] += np.nansum(smahs_fits[masksel], axis=0)
                sfr_data_mean[i] += np.nansum(sfrh[masksel], axis=0)
                sfr_fit_mean[i] += np.nansum(sfrh_fits[masksel], axis=0)

                ngals[i] += masksel.sum()

    mstar_data_mean /= ngals[:, None]
    mstar_fit_mean /= ngals[:, None]
    sfr_data_mean /= ngals[:, None]
    sfr_fit_mean /= ngals[:, None]

    out = (
        mpeak_bins,
        mpeak_binsc,
        tng_t,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    )

    return out


def calculate_plot_galcus_insitu(mpeak_bins):
    BEBOP_GALAC = smhm_utils_galacticus_mgash.BEBOP_GALAC
    BEBOP_GALAC_SFH = smhm_utils_galacticus_mgash.BEBOP_GALAC_SFH

    mpeak_binsc = 0.5 * (mpeak_bins[1:] + mpeak_bins[:-1])

    out = load_galacticus_diffstar_data(BEBOP_GALAC)
    galcus_t = out.galcus_sfh_data["tarr"]
    sfrh = out.galcus_sfh_data["sfh_in_situ"]
    diffmah_data = out.diffmah_fit_data

    log_smahs = np.log10(cumulative_mstar_formed_galpop(galcus_t, sfrh))

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    mah_pars_ntuple = DiffmahParams(*mah_params)
    dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, galcus_t, LGT0)
    logmp0 = log_mah_fit[:, -1]

    # (halo_ids, log_smahs, sfrh, tng_t, log_mahs, logmp0) = out
    log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)
    nt = len(galcus_t)

    mstar_data_mean = np.zeros((len(mpeak_binsc), nt))
    mstar_fit_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_data_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_fit_mean = np.zeros((len(mpeak_binsc), nt))

    ngals = np.zeros(len(mpeak_binsc))

    sfh_type = "in_situ"

    out = smhm_utils_galacticus_mgash.load_diffstar_sfh_tables(
        sfh_type,
        BEBOP_GALAC,
        BEBOP_GALAC_SFH,
    )
    (
        t_table,
        log_mah_table,
        log_smh_table,
        log_ssfrh_table,
        mah_params,
        sfh_params,
        is_cen,
        has_fit,
    ) = out

    log_sfh_table = log_ssfrh_table + log_smh_table

    _log_smahs_data = log_smahs[has_fit]
    _log_sfrh_data = log_sfrh[has_fit]

    _log_smahs_fits = jnp_interp_vmap(galcus_t, t_table, log_smh_table)
    _log_sfrh_fits = jnp_interp_vmap(galcus_t, t_table, log_sfh_table)

    smahs_fits = np.where(_log_smahs_fits == 0.0, np.nan, 10**_log_smahs_fits)
    sfrh_fits = np.where(_log_sfrh_fits == 0.0, np.nan, 10**_log_sfrh_fits)
    smahs_data = np.where(_log_smahs_data == 0.0, np.nan, 10**_log_smahs_data)
    sfrh_data = np.where(_log_sfrh_data == 0.0, np.nan, 10**_log_sfrh_data)

    logmp0_data = logmp0[has_fit]

    ssfrh = sfrh_data / smahs_data
    ssfrh_fit = sfrh_fits / smahs_fits
    ssfrh = np.clip(ssfrh, 1e-12, np.inf)
    ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
    sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
    sfrh_fits = ssfrh_fit * smahs_fits

    for i in range(len(mpeak_bins) - 1):
        masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mstar_data_mean[i] += np.nansum(smahs_data[masksel], axis=0)
            mstar_fit_mean[i] += np.nansum(smahs_fits[masksel], axis=0)
            sfr_data_mean[i] += np.nansum(sfrh[masksel], axis=0)
            sfr_fit_mean[i] += np.nansum(sfrh_fits[masksel], axis=0)

            ngals[i] += masksel.sum()

    mstar_data_mean /= ngals[:, None]
    mstar_fit_mean /= ngals[:, None]
    sfr_data_mean /= ngals[:, None]
    sfr_fit_mean /= ngals[:, None]

    out = (
        mpeak_bins,
        mpeak_binsc,
        galcus_t,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    )

    return out


def calculate_plot_galcus_inplusexsitu(mpeak_bins):
    BEBOP_GALAC = smhm_utils_galacticus_mgash.BEBOP_GALAC
    BEBOP_GALAC_SFH = smhm_utils_galacticus_mgash.BEBOP_GALAC_SFH

    mpeak_binsc = 0.5 * (mpeak_bins[1:] + mpeak_bins[:-1])

    out = load_galacticus_diffstar_data(BEBOP_GALAC)
    galcus_t = out.galcus_sfh_data["tarr"]
    sfrh = out.galcus_sfh_data["sfh_tot"]
    diffmah_data = out.diffmah_fit_data

    log_smahs = np.log10(cumulative_mstar_formed_galpop(galcus_t, sfrh))

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    mah_pars_ntuple = DiffmahParams(*mah_params)
    dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, galcus_t, LGT0)
    logmp0 = log_mah_fit[:, -1]

    # (halo_ids, log_smahs, sfrh, tng_t, log_mahs, logmp0) = out
    log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)
    nt = len(galcus_t)

    mstar_data_mean = np.zeros((len(mpeak_binsc), nt))
    mstar_fit_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_data_mean = np.zeros((len(mpeak_binsc), nt))
    sfr_fit_mean = np.zeros((len(mpeak_binsc), nt))

    ngals = np.zeros(len(mpeak_binsc))

    sfh_type = "in_plus_ex_situ"

    out = smhm_utils_galacticus_mgash.load_diffstar_sfh_tables(
        sfh_type,
        BEBOP_GALAC,
        BEBOP_GALAC_SFH,
    )
    (
        t_table,
        log_mah_table,
        log_smh_table,
        log_ssfrh_table,
        mah_params,
        sfh_params,
        is_cen,
        has_fit,
    ) = out

    log_sfh_table = log_ssfrh_table + log_smh_table

    _log_smahs_data = log_smahs[has_fit]
    _log_sfrh_data = log_sfrh[has_fit]

    _log_smahs_fits = jnp_interp_vmap(galcus_t, t_table, log_smh_table)
    _log_sfrh_fits = jnp_interp_vmap(galcus_t, t_table, log_sfh_table)

    smahs_fits = np.where(_log_smahs_fits == 0.0, np.nan, 10**_log_smahs_fits)
    sfrh_fits = np.where(_log_sfrh_fits == 0.0, np.nan, 10**_log_sfrh_fits)
    smahs_data = np.where(_log_smahs_data == 0.0, np.nan, 10**_log_smahs_data)
    sfrh_data = np.where(_log_sfrh_data == 0.0, np.nan, 10**_log_sfrh_data)

    logmp0_data = logmp0[has_fit]

    ssfrh = sfrh_data / smahs_data
    ssfrh_fit = sfrh_fits / smahs_fits
    ssfrh = np.clip(ssfrh, 1e-12, np.inf)
    ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
    sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
    sfrh_fits = ssfrh_fit * smahs_fits

    for i in range(len(mpeak_bins) - 1):
        masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mstar_data_mean[i] += np.nansum(smahs_data[masksel], axis=0)
            mstar_fit_mean[i] += np.nansum(smahs_fits[masksel], axis=0)
            sfr_data_mean[i] += np.nansum(sfrh[masksel], axis=0)
            sfr_fit_mean[i] += np.nansum(sfrh_fits[masksel], axis=0)

            ngals[i] += masksel.sum()

    mstar_data_mean /= ngals[:, None]
    mstar_fit_mean /= ngals[:, None]
    sfr_data_mean /= ngals[:, None]
    sfr_fit_mean /= ngals[:, None]

    out = (
        mpeak_bins,
        mpeak_binsc,
        galcus_t,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    )

    return out


def save_data(outdrn, outname, data):
    fnout = os.path.join(outdrn, outname)

    (
        tarr,
        log_smahs_fits,
        log_sfrh_fits,
        log_smahs_data,
        log_sfrh_data,
        logmp0_data,
    ) = data

    with h5py.File(fnout, "w") as hdfout:
        hdfout["tarr"] = tarr
        hdfout["log_smahs_fits"] = log_smahs_fits
        hdfout["log_sfrh_fits"] = log_sfrh_fits
        hdfout["log_smahs_data"] = log_smahs_data
        hdfout["log_sfrh_data"] = log_sfrh_data
        hdfout["logmp0_data"] = logmp0_data


def save_data_plot(outdrn, outname, data):
    fnout = os.path.join(outdrn, outname)

    (
        mpeak_bins,
        mpeak_binsc,
        tarr,
        mstar_data_mean,
        mstar_fit_mean,
        sfr_data_mean,
        sfr_fit_mean,
    ) = data

    with h5py.File(fnout, "w") as hdfout:
        hdfout["tarr"] = tarr
        hdfout["mpeak_bins"] = mpeak_bins
        hdfout["mpeak_binsc"] = mpeak_binsc
        hdfout["mstar_data_mean"] = mstar_data_mean
        hdfout["mstar_fit_mean"] = mstar_fit_mean
        hdfout["sfr_data_mean"] = sfr_data_mean
        hdfout["sfr_fit_mean"] = sfr_fit_mean


# out_smdpl_nomerging = calculate_smdpl_nomerging()
# outdir = "/lcrc/project/halotools/alarcon/results/diffstar_quality_fits/"
# outname = "diffstar_quality_smdpl.h5"
# save_data(outdir, outname, out_smdpl_nomerging)

# outdir = "/lcrc/project/halotools/alarcon/results/smdpl_pdf_target_data/"
# sim_name = "SMDPL_UM_Nomerging"
# make_diffstar_fits_plot(outdir, sim_name, *out_smdpl_nomerging)

# mpeak_bins = np.arange(11.25, 14.5, 0.50)
# out_smdpl_nomerging = calculate_plot_smdpl_nomerging(mpeak_bins)
# outdir = "/lcrc/project/halotools/alarcon/results/diffstar_quality_fits/"
# outname = "diffstar_quality_smdpl.h5"
# save_data_plot(outdir, outname, out_smdpl_nomerging)

# mpeak_bins = np.arange(11.25, 14.5, 0.50)
# out_smdpl_dr1 = calculate_plot_smdpl_dr1(mpeak_bins)
# outdir = "/lcrc/project/halotools/alarcon/results/diffstar_quality_fits/"
# outname = "diffstar_quality_smdpl_dr1.h5"
# save_data_plot(outdir, outname, out_smdpl_dr1)

# mpeak_bins = np.arange(11.25, 14.5, 0.50)
# out_tng = calculate_plot_tng(mpeak_bins)
# outdir = "/lcrc/project/halotools/alarcon/results/diffstar_quality_fits/"
# outname = "diffstar_quality_tng.h5"
# save_data_plot(outdir, outname, out_tng)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sim_name",
        help="simulation name",
        type=str,
        default="smdpl",
        choices=[
            "smdpl",
            "smdpl_dr1",
            "tng",
            "galcus_insitu",
            "galcus_inplusexsitu",
            "all",
        ],
    )
    parser.add_argument(
        "-outdir",
        help="simulation name",
        type=str,
        default="/lcrc/project/halotools/alarcon/results/mgash/diffstar_quality_fits_mgash/",
    )

    args = parser.parse_args()
    sim_name = args.sim_name
    outdir = args.outdir

    mpeak_bins = np.arange(11.25, 14.5, 0.50)

    if sim_name == "all":
        out_smdpl_nomerging = calculate_plot_smdpl_nomerging(mpeak_bins)
        outname = "diffstar_quality_smdpl.h5"
        save_data_plot(outdir, outname, out_smdpl_nomerging)

        out_smdpl_dr1 = calculate_plot_smdpl_dr1(mpeak_bins)
        outname = "diffstar_quality_smdpl_dr1.h5"
        save_data_plot(outdir, outname, out_smdpl_dr1)

        out_tng = calculate_plot_tng(mpeak_bins)
        outname = "diffstar_quality_tng.h5"
        save_data_plot(outdir, outname, out_tng)

        out_galcus_insitu = calculate_plot_galcus_insitu(mpeak_bins)
        outname = "diffstar_quality_galcus_insitu.h5"
        save_data_plot(outdir, outname, out_galcus_insitu)

        out_galcus_inplusexsitu = calculate_plot_galcus_inplusexsitu(mpeak_bins)
        outname = "diffstar_quality_galcus_inplusexsitu.h5"
        save_data_plot(outdir, outname, out_galcus_inplusexsitu)

    elif sim_name == "smdpl":
        out_smdpl_nomerging = calculate_plot_smdpl_nomerging(mpeak_bins)
        outname = "diffstar_quality_smdpl.h5"
        save_data_plot(outdir, outname, out_smdpl_nomerging)
    elif sim_name == "smdpl_dr1":
        out_smdpl_dr1 = calculate_plot_smdpl_dr1(mpeak_bins)
        outname = "diffstar_quality_smdpl_dr1.h5"
        save_data_plot(outdir, outname, out_smdpl_dr1)
    elif sim_name == "tng":
        out_tng = calculate_plot_tng(mpeak_bins)
        outname = "diffstar_quality_tng.h5"
        save_data_plot(outdir, outname, out_tng)
    elif sim_name == "galcus_insitu":
        out_galcus_insitu = calculate_plot_galcus_insitu(mpeak_bins)
        outname = "diffstar_quality_galcus_insitu.h5"
        save_data_plot(outdir, outname, out_galcus_insitu)
    elif sim_name == "galcus_inplusexsitu":
        out_galcus_inplusexsitu = calculate_plot_galcus_inplusexsitu(mpeak_bins)
        outname = "diffstar_quality_galcus_inplusexsitu.h5"
        save_data_plot(outdir, outname, out_galcus_inplusexsitu)
