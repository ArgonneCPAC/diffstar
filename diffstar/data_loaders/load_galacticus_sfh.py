""" """

import os
from collections import namedtuple

import h5py
import numpy as np

from ..utils import _get_dt_array
from . import load_flat_hdf5

DRN_LCRC = "/lcrc/project/halotools/Galacticus/diffstarpop_data"
DRN_POBOY = "/Users/aphearin/work/DATA/Galacticus/diffstarpop_data"
BN_DIFFSTAR_IN_SITU = "diffstar_fits_in_situ.hdf5"
BN_DIFFSTAR_IN_PLUS_EX_SITU = "diffstar_fits_in_plus_ex_situ.hdf5"
BN_DIFFMAH = "diffmah_fits.h5"
BN_GALCUS = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"
BN_GALCUS_REDUCTION = "sfh_disk_bulge_in_ex_situ.hdf5"
BNAME_SFH_DATA = "sfh_disk_bulge_in_ex_situ.hdf5"


# Planck18 cosmology assumed in BNAME_APH2
Ob0 = 0.04930
Om0 = 0.31530
FB = Ob0 / Om0


def load_galacticus_diffstar_data(drn):
    """Load data used in DiffstarPop analysis of Galacticus SFH

    Parameters
    ----------
    drn : string
        Directory storing results

    Returns
    -------
    diffstar_fit_data : namedtuple, with fields:

        diffmah_fit_data : dict
            Columns store the diffmah fitter results

        diffstar_in_situ_fit_data : dict
            Columns store the diffstar fitter results

        galcus_sfh_data : dict
            Columns are ('tarr', 'sfh_in_situ', 'sfh_tot', 'is_cen')

    """
    fn_diffmah = os.path.join(drn, BN_DIFFMAH)
    diffmah_fit_data = load_flat_hdf5(fn_diffmah)

    fn_diffstar = os.path.join(drn, BN_DIFFSTAR_IN_SITU)
    diffstar_in_situ_fit_data = load_flat_hdf5(fn_diffstar)

    fn_diffstar_tot_sfh = os.path.join(drn, BN_DIFFSTAR_IN_PLUS_EX_SITU)
    diffstar_in_plus_ex_situ_fit_data = load_flat_hdf5(fn_diffstar_tot_sfh)

    galcus_sfh_data = load_galacticus_sfh_target_data(drn)

    ret_colnames = (
        "diffmah_fit_data",
        "diffstar_in_situ_fit_data",
        "diffstar_in_plus_ex_situ_fit_data",
        "galcus_sfh_data",
    )
    DiffstarFitData = namedtuple("DiffstarFitData", ret_colnames)
    diffstar_fit_data = DiffstarFitData(
        diffmah_fit_data,
        diffstar_in_situ_fit_data,
        diffstar_in_plus_ex_situ_fit_data,
        galcus_sfh_data,
    )
    return diffstar_fit_data


def load_galacticus_sfh_target_data(drn):
    galcus_sfh_data = dict()
    galcus_sfh_data["tarr"] = np.load(os.path.join(drn, "tarr_bulge.npy"))
    dtarr = np.diff(np.concatenate(([0.0], galcus_sfh_data["tarr"])))

    # raw SFH data gives ΔMstar(Δt) in units of ΔMsun and ΔGyr, so we divide by Δt*1e9
    dmstar_in_situ_bulge = np.load(os.path.join(drn, "delta_mstar_in_situ_bulge.npy"))
    dmstar_in_situ_disk = np.load(os.path.join(drn, "delta_mstar_in_situ_disk.npy"))
    dmstar_in_situ = dmstar_in_situ_bulge + dmstar_in_situ_disk
    sfh_in_situ = dmstar_in_situ / dtarr / 1e9

    dmstar_tot_bulge = np.load(os.path.join(drn, "delta_mstar_tot_bulge.npy"))
    dmstar_tot_disk = np.load(os.path.join(drn, "delta_mstar_tot_disk.npy"))
    dmstar_tot = dmstar_tot_bulge + dmstar_tot_disk
    sfh_tot = dmstar_tot / dtarr / 1e9

    galcus_sfh_data["sfh_in_situ"] = sfh_in_situ
    galcus_sfh_data["sfh_tot"] = sfh_tot

    fn_galcus = os.path.join(drn, BN_GALCUS)
    with h5py.File(fn_galcus, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]
        nodeIsIsolated = nodeData["nodeIsIsolated"][:]

    galcus_sfh_data["is_cen"] = nodeIsIsolated

    return galcus_sfh_data


def write_galacticus_sfh_data_block(drn):
    galcus_sfh_data = load_galacticus_sfh_target_data(drn)
    fnout = os.path.join(drn, BNAME_SFH_DATA)
    with h5py.File(fnout, "w") as hdf:
        hdf["sfh_in_situ"] = galcus_sfh_data["sfh_in_situ"]
        hdf["sfh_tot"] = galcus_sfh_data["sfh_tot"]
        hdf["is_cen"] = galcus_sfh_data["is_cen"]
        hdf["tarr"] = galcus_sfh_data["tarr"]


def load_galacticus_sfh_data_block(fn, istart, iend):
    tarr_dict = load_flat_hdf5(fn, keys=["tarr"])
    block_keys = ("sfh_in_situ", "sfh_tot", "is_cen")
    sfh_data_block = load_flat_hdf5(fn, istart=istart, iend=iend, keys=block_keys)
    sfh_data_block["tarr"] = tarr_dict["tarr"]

    return sfh_data_block
