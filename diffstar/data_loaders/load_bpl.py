"""
"""
import os
from collections import OrderedDict

import h5py
import numpy as np

try:
    from astropy.table import Table

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

from ..kernels.main_sequence_kernels import _get_bounded_sfr_params_vmap
from ..kernels.quenching_kernels import _get_bounded_q_params_vmap
from ..utils import _jax_get_dt_array

LGT0 = 1.13980

TASSO_BPL_DRN = "/Users/aphearin/work/DATA/diffstar_data"
BEBOP_BPL_DRN = "/lcrc/project/halotools/diffstar_data"

MAH_COLNAMES = [
    "mah_fit_logmp",
    "mah_fit_logtc",
    "mah_fit_early_index",
    "mah_fit_late_index",
]
MS_U_COLNAMES = [
    "sfh_fit_u_lgmcrit",
    "sfh_fit_u_lgy_at_mcrit",
    "sfh_fit_u_indx_lo",
    "sfh_fit_u_indx_hi",
    "sfh_fit_u_tau_dep",
]
Q_U_COLNAMES = [
    "sfh_fit_u_qt",
    "sfh_fit_u_qs",
    "sfh_fit_u_q_drop",
    "sfh_fit_u_q_rejuv",
]


def load_bpl_diffstar_data(
    drn,
    mah_colnames=MAH_COLNAMES,
    ms_u_colnames=MS_U_COLNAMES,
    q_u_colnames=Q_U_COLNAMES,
):
    assert HAS_ASTROPY, "Must have astropy installed to use load_bpl_diffstar_data"
    bpl_fn = os.path.join(drn, "bpl_diffmah_cens.npy")
    trunks = np.load(bpl_fn)

    diffmah_fn = os.path.join(drn, "run1_bpl_diffmah.h5")
    diffmah_fits = OrderedDict()
    with h5py.File(diffmah_fn) as hdf:
        for key in hdf.keys():
            diffmah_fits[key] = hdf[key][...]

    diffstar_fn = os.path.join(drn, "bpl_diffstar_fits_default.h5")
    diffstar_fits = OrderedDict()
    with h5py.File(diffstar_fn) as hdf:
        for key in hdf.keys():
            diffstar_fits[key] = hdf[key][...]

    pat = "{0} and {1} have mismatched halo_id column"
    msg = pat.format(bpl_fn, diffmah_fn)
    assert np.allclose(trunks["halo_id"], diffmah_fits["halo_id"]), msg

    msg = pat.format(bpl_fn, diffstar_fn)
    assert np.allclose(trunks["halo_id"], diffstar_fits["halo_id"]), msg

    bpl = Table()
    bpl["halo_id"] = trunks["halo_id"]
    bpl["upid"] = trunks["upid"]
    bpl["sfrh_sim"] = trunks["sfr_history_main_prog"]

    time_fn = os.path.join(drn, "bpl_cosmic_time.npy")
    t_bpl = np.load(time_fn)
    dt_bpl = _jax_get_dt_array(t_bpl)
    log_mahs, log_smh = _compute_logmah_logsmh(
        dt_bpl, trunks["mpeak_history_main_prog"], trunks["sfr_history_main_prog"]
    )
    bpl["log_mah_sim"] = log_mahs
    bpl["logsmh_sim"] = log_smh

    mah_keys = ["logmp_fit", "mah_logtc", "mah_k", "early_index", "late_index", "loss"]
    for key in mah_keys:
        if key[:4] != "mah_":
            newkey = "mah_fit_" + key
        else:
            newkey = "mah_fit_" + key[4:]
        bpl[newkey] = diffmah_fits[key]
    bpl.rename_column("mah_fit_logmp_fit", "mah_fit_logmp")

    diffstar_param_keys = [
        "u_lgmcrit",
        "u_lgy_at_mcrit",
        "u_indx_lo",
        "u_indx_hi",
        "u_tau_dep",
        "u_qt",
        "u_qs",
        "u_q_drop",
        "u_q_rejuv",
        "loss",
        "success",
    ]
    for key in diffstar_param_keys:
        bpl["sfh_fit_" + key] = diffstar_fits[key]

    u_ms_colnames = [
        "sfh_fit_u_lgmcrit",
        "sfh_fit_u_lgy_at_mcrit",
        "sfh_fit_u_indx_lo",
        "sfh_fit_u_indx_hi",
        "sfh_fit_u_tau_dep",
    ]
    ms_u_params_list = [bpl[key] for key in ms_u_colnames]

    q_u_params_list = [bpl[key] for key in q_u_colnames]

    ms_u_params = np.array(ms_u_params_list).T
    bounded_ms_params = _get_bounded_sfr_params_vmap(ms_u_params).T

    q_u_params = np.array(q_u_params_list).T
    bounded_q_params = _get_bounded_q_params_vmap(q_u_params).T

    u_ms_colnames_v0p2 = u_ms_colnames
    for key, arr in zip(u_ms_colnames_v0p2, bounded_ms_params):
        newkey = key.replace("_u_", "_")
        bpl[newkey] = arr

    u_q_colnames_v0p2 = (
        "sfh_fit_u_lg_qt",
        "sfh_fit_u_qlglgdt",
        "sfh_fit_u_lg_drop",
        "sfh_fit_u_lg_rejuv",
    )
    for key, arr in zip(u_q_colnames_v0p2, bounded_q_params):
        newkey = key.replace("_u_", "_")
        bpl[newkey] = arr

    bpl.rename_column("sfh_fit_u_qt", "sfh_fit_u_lg_qt")
    bpl.rename_column("sfh_fit_u_qs", "sfh_fit_u_qlglgdt")
    bpl.rename_column("sfh_fit_u_q_drop", "sfh_fit_u_lg_drop")
    bpl.rename_column("sfh_fit_u_q_rejuv", "sfh_fit_u_lg_rejuv")

    all_param_colnames = mah_colnames, u_ms_colnames_v0p2, u_q_colnames_v0p2
    return bpl, t_bpl, all_param_colnames


def _compute_logmah_logsmh(dt, mah, sfrh):
    smh = np.cumsum(sfrh * dt, axis=1) * 1e9
    mah = np.maximum.accumulate(mah, axis=1)

    EPS = 0.01
    mah = np.where(mah <= EPS, EPS, mah)
    smh = np.where(smh <= EPS, EPS, smh)

    log_mahs = np.log10(mah)
    log_smh = np.log10(smh)

    return log_mahs, log_smh
