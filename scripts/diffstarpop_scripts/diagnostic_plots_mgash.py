import os
import h5py
import numpy as np
import jax.numpy as jnp
import argparse

from collections import OrderedDict, namedtuple

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl

from diffstar.defaults import TODAY, LGT0
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop

from diffstar.sfh_model_mgash import _cumulative_mstar_formed_vmap

from diffstar.diffstarpop.mc_diffstarpop_mgash import (
    mc_diffstar_sfh_galpop,
)
from diffstar.diffstarpop.kernels.defaults_mgash import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
    get_bounded_diffstarpop_params,
)
from fit_get_loss_helpers_mgash_anyz import (
    get_loss_data_smhm,
    get_loss_data_pdfs_mstar,
    get_loss_data_pdfs_ssfr_central,
    get_loss_data_pdfs_ssfr_satellite,
)

from diffstar.diffstarpop.loss_kernels.mstar_ssfr_loss_mgash_anyz import (
    get_pred_mstar_data_wrapper,
    get_pred_mstar_ssfr_data_wrapper,
    get_pred_mstar_ssfr_sat_data_wrapper,
)

import subprocess


def try_enable_latex():
    """Try enabling LaTeX text rendering in matplotlib,
    fallback if not available."""
    try:
        # Quick check: can we run latex?
        subprocess.check_call(
            ["latex", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif", size=16)
        print("LaTeX rendering enabled.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # LaTeX not installed or failed
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif", size=16)
        print("LaTeX not available, falling back to default mathtext.")


BEBOP_SMHM_MEAN_DATA = "/lcrc/project/halotools/alarcon/results/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-indir", help="input drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "-outdir", help="output drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="Path were diffstarpop params are stored",
    )
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    params_path = args.params_path

    # Quick check: can we run latex?
    try_enable_latex()

    if params_path is None:
        all_u_params = jnp.array(DEFAULT_DIFFSTARPOP_U_PARAMS)
    else:
        params = np.load(params_path)
        all_u_params = params["diffstarpop_u_params"]

    # Define params ---------------------------------------------
    diffstarpop_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._make(all_u_params)

    diffstarpop_params = get_bounded_diffstarpop_params(diffstarpop_u_params)

    # Load SMHM data ---------------------------------------------
    nhalos = 10
    _, plot_data_SMHM = get_loss_data_smhm(indir, nhalos)
    _, plot_data_pdf = get_loss_data_pdfs_mstar(indir, nhalos)
    _, plot_data_pdf_ssfr_cen = get_loss_data_pdfs_ssfr_central(indir, nhalos)
    _, plot_data_pdf_ssfr_sat = get_loss_data_pdfs_ssfr_satellite(indir, nhalos)

    # ============
    # Plots SMHM
    # ============
    print("Making SMHM plot...")

    (
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
    ) = plot_data_SMHM

    mstar_plot = np.zeros((len(age_targets), len(logmh_binsc)))
    mstar_plot_grad = np.zeros((len(age_targets), len(logmh_binsc)))

    smhm_plot = smhm.copy()

    for i in range(len(age_targets)):
        t_target = age_targets[i]
        print("Age target:", t_target)
        tarr = np.logspace(-1, np.log10(t_target), 50)

        for j in range(len(logmh_binsc)):

            sel = (tobs_id == i) & (logmh_id == j)
            if sel.sum() < 50:
                smhm_plot[i, j] = np.nan
                mstar_plot[i, j] = np.nan
                continue

            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            lomg0_vals = log_mah_fit[:, -1]
            upid_vals = upid_samp[sel]
            res = mc_diffstar_sfh_galpop(
                diffstarpop_params,
                mah_params_samp[:, sel],
                lomg0_vals,
                upid_vals,
                np.ones(sel.sum()) * lgmu_infall,
                np.ones(sel.sum()) * logmhost_infall,
                np.ones(sel.sum()) * gyr_since_infall,
                ran_key,
                tarr,
            )

            (
                diffstar_params_ms,
                diffstar_params_q,
                sfh_ms,
                sfh_q,
                frac_q,
                mc_is_q,
            ) = res
            mstar_q = _cumulative_mstar_formed_vmap(tarr, sfh_q)
            mstar_ms = _cumulative_mstar_formed_vmap(tarr, sfh_ms)
            mean_mstar_grad_vals = mstar_q[:, -1] * frac_q + mstar_ms[:, -1] * (
                1 - frac_q
            )
            mean_mstar_grad = jnp.mean(jnp.log10(mean_mstar_grad_vals))

            mean_mstar_plot_vals = mstar_q[:, -1] * mc_is_q.astype(int).astype(
                float
            ) + mstar_ms[:, -1] * (1.0 - mc_is_q.astype(int))
            mean_mstar_plot_vals = jnp.mean(jnp.log10(mean_mstar_plot_vals))
            mstar_plot[i, j] = mean_mstar_plot_vals
            mstar_plot_grad[i, j] = mean_mstar_grad
            # break

            cmap = plt.get_cmap("plasma")(redshift_targets / redshift_targets.max())

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(len(smhm)):
        ax.plot(10**logmh_binsc, 10 ** smhm_plot[i], color=cmap[i])
        ax.plot(10**logmh_binsc, 10 ** mstar_plot[i], color=cmap[i], ls="--")

    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma),
        ax=ax,
        label="Redshift",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(9e10, 1e15)

    ax.set_ylabel(
        r"$\langle M_\star(t_{\rm obs})| M_{\rm halo}(t_{\rm obs}) \rangle$ $[M_\odot]$"
    )
    ax.set_xlabel(r"$M_{\rm halo}(t_{\rm obs})$ $[M_\odot]$")
    plt.savefig(outdir + "smhm_logsm.png", bbox_inches="tight", dpi=300)
    plt.clf()

    # =====================================
    # Plots for P(Mstar | Mobs, zobs)
    # =====================================
    print("Making plot Mstar PDFs...")
    fontsize = 24
    tick_fontsize = 24
    (
        logmstar_bins_pdf,
        mstar_wcounts,
        age_targets,
        redshift_targets,
        tobs_id,
        logmh_id,
        logmh_binsc,
        loss_data_mstar_pred,
    ) = plot_data_pdf

    _mstar_counts_pred = get_pred_mstar_data_wrapper(all_u_params, loss_data_mstar_pred)

    mstar_counts_pred = np.zeros_like(mstar_wcounts) * np.nan
    ij = 0
    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            # if sel.sum() == 0: continue
            if sel.sum() < 50:
                continue
            mstar_counts_pred[i, j] = _mstar_counts_pred[ij]
            ij += 1

    fig, ax = plt.subplots(
        len(age_targets), 1, figsize=(12, 16 * len(age_targets) / 5), sharex=False
    )

    colors_mstar = plt.get_cmap("viridis")(np.linspace(0, 1, 11))

    for i in range(len(age_targets)):
        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)
            if sel.sum() < 50:
                continue
            ax[i].fill_between(
                logmstar_bins_pdf[1:],
                0.0,
                mstar_wcounts[i, j] / mstar_wcounts[i, j].sum(),
                color=colors_mstar[j],
                alpha=0.2,
            )
            ax[i].plot(
                logmstar_bins_pdf[1:],
                mstar_counts_pred[i, j],
                color=colors_mstar[j],
                ls="--",
            )

        ax[i].set_ylim(0, 0.35)
        ax[i].set_yticks(np.arange(0, 0.31, 0.10))
        ax[i].set_xlim(7, 13.0)
        ax[i].set_ylabel(r"$P(M_{\star}| M_{\rm halo})$", fontsize=fontsize)
        ax[i].set_title(
            r"${\rm Redshift}=%.1f$" % redshift_targets[i],
            y=0.82,
            x=0.88,
            fontsize=fontsize,
        )
        if i < len(age_targets) - 1:
            ax[i].set_xticklabels([])
        ax[i].yaxis.set_tick_params(labelsize=tick_fontsize)

    legend_elements = [
        Patch(
            facecolor=colors_mstar[0],
            edgecolor="none",
            label=r"$m_{\rm h}(t_{\rm obs})=11$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_mstar[3],
            edgecolor="none",
            label=r"$m_{\rm h}(t_{\rm obs})=12$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_mstar[6],
            edgecolor="none",
            label=r"$m_{\rm h}(t_{\rm obs})=13$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_mstar[-2],
            edgecolor="none",
            label=r"$m_{\rm h}(t_{\rm obs})=14$",
            alpha=0.7,
        ),
        Line2D([0], [0], color="k", ls="--", label="Diffstarpop"),
    ]
    ax[0].legend(handles=legend_elements, loc=2, fontsize=18)
    ax[len(age_targets) - 1].set_xlabel(
        r"$\log M_\star(t_{\rm obs}) [M_\odot]$", fontsize=fontsize
    )

    ax[len(age_targets) - 1].xaxis.set_tick_params(labelsize=tick_fontsize)

    fig.subplots_adjust(hspace=0.08)

    plt.savefig(
        outdir + "pdf_mstar.png",
        bbox_inches="tight",
        dpi=250,
    )
    plt.clf()

    # =================================================
    # Plots P(sSFR | Mstar, zobs) for centrals
    # =================================================
    print("Making plot sSFR PDFs for centrals...")

    (
        target_mstar_ids,
        logssfr_binsc_pdf,
        target_data,
        loss_data_ssfr_pred,
    ) = plot_data_pdf_ssfr_cen

    bestfit_data = get_pred_mstar_ssfr_data_wrapper(all_u_params, loss_data_ssfr_pred)

    fig, ax = plt.subplots(
        len(age_targets), 1, figsize=(12, 16 * len(age_targets) / 5), sharex=False
    )

    colors_ssfr = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(target_mstar_ids)))

    for i in range(len(age_targets)):

        for j in range(len(target_mstar_ids)):

            ax[i].fill_between(
                logssfr_binsc_pdf,
                0.0,
                target_data[i, j],
                color=colors_ssfr[j],
                alpha=0.2,
            )
            ax[i].plot(
                logssfr_binsc_pdf, bestfit_data[i, j], color=colors_ssfr[j], ls="--"
            )

        ax[i].set_ylim(0, 0.37)
        ax[i].set_yticks(np.arange(0, 0.31, 0.10))
        ax[i].set_ylabel(r"$P_{\rm cen}({\rm sSFR}| M_\star)$", fontsize=fontsize)
        ax[i].set_title(
            r"${\rm Redshift}=%.1f$" % redshift_targets[i],
            y=0.82,
            x=0.12,
            fontsize=fontsize,
        )
        if i < len(age_targets) - 1:
            ax[i].set_xticklabels([])
        ax[i].yaxis.set_tick_params(labelsize=tick_fontsize)

    legend_elements = [
        Patch(
            facecolor=colors_ssfr[0],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=9.0$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_ssfr[2],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=10.0$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_ssfr[-1],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=11.5$",
            alpha=0.7,
        ),
        Line2D([0], [0], color="k", ls="--", label="Diffstarpop"),
    ]
    ax[0].legend(handles=legend_elements, loc=1, fontsize=18)
    ax[len(age_targets) - 1].set_xlabel(
        r"$\log {\rm sSFR}(t_{\rm obs}) [\rm{yr}^{-1}]$", fontsize=fontsize
    )
    ax[len(age_targets) - 1].xaxis.set_tick_params(labelsize=tick_fontsize)

    fig.subplots_adjust(hspace=0.08)
    fig.suptitle("Centrals", y=0.9, fontsize=fontsize)
    plt.savefig(
        outdir + "pdf_ssfr_centrals.png",
        bbox_inches="tight",
        dpi=250,
    )
    plt.clf()

    # =================================================
    # Plots P(sSFR | Mstar, zobs) for satellites
    # =================================================

    print("Making plot sSFR PDFs for centrals...")

    (
        target_mstar_ids,
        logssfr_binsc_pdf,
        target_data_sat,
        loss_data_ssfr_sat_pred,
    ) = plot_data_pdf_ssfr_sat

    bestfit_data = get_pred_mstar_ssfr_sat_data_wrapper(
        all_u_params, loss_data_ssfr_sat_pred
    )

    fig, ax = plt.subplots(
        len(age_targets), 1, figsize=(12, 16 * len(age_targets) / 5), sharex=False
    )

    colors_ssfr = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(target_mstar_ids)))

    for i in range(len(age_targets)):

        for j in range(len(target_mstar_ids)):

            ax[i].fill_between(
                logssfr_binsc_pdf,
                0.0,
                target_data_sat[i, j],
                color=colors_ssfr[j],
                alpha=0.2,
            )
            ax[i].plot(
                logssfr_binsc_pdf, bestfit_data[i, j], color=colors_ssfr[j], ls="--"
            )

        ax[i].set_ylim(0, 0.37)
        ax[i].set_yticks(np.arange(0, 0.31, 0.10))
        ax[i].set_ylabel(r"$P_{\rm sat}({\rm sSFR}| M_\star)$", fontsize=fontsize)
        ax[i].set_title(
            r"${\rm Redshift}=%.1f$" % redshift_targets[i],
            y=0.82,
            x=0.12,
            fontsize=fontsize,
        )
        if i < len(age_targets) - 1:
            ax[i].set_xticklabels([])
        ax[i].yaxis.set_tick_params(labelsize=tick_fontsize)

    legend_elements = [
        Patch(
            facecolor=colors_ssfr[0],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=9.0$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_ssfr[2],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=10.0$",
            alpha=0.7,
        ),
        Patch(
            facecolor=colors_ssfr[-1],
            edgecolor="none",
            label=r"$m_\star(t_{\rm obs})=11.5$",
            alpha=0.7,
        ),
        Line2D([0], [0], color="k", ls="--", label="Diffstarpop"),
    ]
    ax[0].legend(handles=legend_elements, loc=1, fontsize=18)
    ax[len(age_targets) - 1].set_xlabel(
        r"$\log {\rm sSFR}(t_{\rm obs}) [\rm{yr}^{-1}]$", fontsize=fontsize
    )

    ax[len(age_targets) - 1].xaxis.set_tick_params(labelsize=tick_fontsize)

    fig.subplots_adjust(hspace=0.08)
    fig.suptitle("Satellites", y=0.9, fontsize=fontsize)

    plt.savefig(
        outdir + "pdf_ssfr_satellites.png",
        bbox_inches="tight",
        dpi=250,
    )
    plt.clf()
