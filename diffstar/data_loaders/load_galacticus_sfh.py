""" """

import os

import h5py

from . import load_flat_hdf5

DRN_LCRC = "/lcrc/project/halotools/Galacticus/diffstarpop_data"
DRN_POBOY = "/Users/aphearin/work/DATA/Galacticus/diffstarpop_data"
BN_DIFFSTAR = "diffstar_fits.hdf5"
BN_DIFFMAH = "diffmah_fits.h5"
BN_GALCUS = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"
BN_GALCUS_REDUCTION = "sfh_disk_bulge_in_ex_situ.hdf5"


def load_galacticus_diffstar_data(drn):
    """Load data used in DiffstarPop analysis of Galacticus SFH

    Parameters
    ----------
    drn : string
        Directory storing results

    Returns
    -------
    diffmah_fit_data : dict
        Columns store the diffmah fitter results

    diffstar_fit_data : dict
        Columns store the diffstar fitter results

    galcus_sfh_data : dict
        Columns are ('tarr', 'sfh_in_situ', 'sfh_tot', 'is_cen')

    """
    fn_diffmah = os.path.join(drn, BN_DIFFMAH)
    diffmah_fit_data = load_flat_hdf5(fn_diffmah)

    fn_diffstar = os.path.join(drn, BN_DIFFSTAR)
    diffstar_fit_data = load_flat_hdf5(fn_diffstar)

    fn_sfh_target_data = os.path.join(drn, BN_GALCUS_REDUCTION)
    raw_sfh_data = load_flat_hdf5(fn_sfh_target_data)

    galcus_sfh_data = dict()
    galcus_sfh_data["tarr"] = raw_sfh_data["tarr"]

    # raw SFH data is in units of Msun/Gyr, so we divide by 1e9
    _raw_in_situ = raw_sfh_data["sfh_in_situ_bulge"] + raw_sfh_data["sfh_in_situ_disk"]
    galcus_sfh_data["sfh_in_situ"] = _raw_in_situ / 1e9
    _raw_tot_sfh = raw_sfh_data["sfh_tot_bulge"] + raw_sfh_data["sfh_tot_disk"]
    galcus_sfh_data["sfh_tot"] = _raw_tot_sfh / 1e9

    fn_galcus = os.path.join(drn, BN_GALCUS)
    with h5py.File(fn_galcus, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]
        nodeIsIsolated = nodeData["nodeIsIsolated"][:]

    galcus_sfh_data["is_cen"] = nodeIsIsolated

    return diffmah_fit_data, diffstar_fit_data, galcus_sfh_data


def load_galacticus_sfh_data_block(fn, istart, iend):
    tarr_dict = load_flat_hdf5(fn, keys=["tarr"])
    block_keys = (
        "sfh_in_situ_bulge",
        "sfh_in_situ_disk",
        "sfh_tot_bulge",
        "sfh_tot_disk",
    )
    sfh_data_block = load_flat_hdf5(fn, istart=istart, iend=iend, keys=block_keys)
    sfh_data_block["tarr"] = tarr_dict["tarr"]

    return sfh_data_block
