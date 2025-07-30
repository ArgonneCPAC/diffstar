""" """

import argparse
import os
from time import time

import h5py
import numpy as np

DRN_ROOT_POBOY = "/Users/aphearin/work/DATA/Galacticus/diffstarpop_data/central"
SUBDRN_APH1 = "scratch/sweeraso/outputs/AHearin"
DRN_POBOY_APH1 = os.path.join(DRN_ROOT_POBOY, SUBDRN_APH1)
BNAME_APH1 = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin_9583trees.hdf5"

DISK_COLNAME = "diskStarFormationHistoryMass"
BULGE_COLNAME = "spheroidStarFormationHistoryMass"

DRN_LCRC_APH2 = "/lcrc/project/halotools/Galacticus/diffstarpop_data"
BNAME_APH2 = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"


def get_nhalos_tot(fn):
    with h5py.File(fn, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]
        nhalos_tot = nodeData["haloAccretionHistoryMass"].size
    return nhalos_tot


def extract_galacticus_mah_tables(fn, istart=0, iend=None):

    with h5py.File(fn, "r") as hdf:
        nodeData = hdf["Outputs"]["Output1"]["nodeData"]

        if iend is None:
            iend = nodeData["haloAccretionHistoryMass"].size

        MAHdata = nodeData["haloAccretionHistoryMass"][istart:iend]
        MAHtimes = nodeData["haloAccretionHistoryTime"][istart:iend]
        nodeIsIsolated = nodeData["nodeIsIsolated"][istart:iend]
        basicTimeLastIsolated = nodeData["basicTimeLastIsolated"][istart:iend]

    return MAHdata, MAHtimes, nodeIsIsolated, basicTimeLastIsolated


def extract_galacticus_sfh_tables(fn, colname, istart=0, iend=None):
    with h5py.File(fn, "r") as hdf:
        SFH_data = hdf["Outputs"]["Output1"]["nodeData"][colname]
        tarr = SFH_data.attrs["time"]

        n_gals_tot = SFH_data.size
        if iend is None:
            iend = n_gals_tot
        n_gals_out = iend - istart

        for _x in SFH_data:
            if _x.size > 0:
                n_met = len(_x) // 2
                n_t = _x[0].size
                break

        delta_mstar_in_situ_table = np.zeros((n_gals_out, n_t))
        delta_mstar_tot_table = np.zeros((n_gals_out, n_t))

        for igal in range(istart, iend):
            sfh_data_igal = SFH_data[igal]

            if len(sfh_data_igal) > 0:
                X_sfh_tot = np.array([sfh_data_igal[j] for j in range(0, n_met)])

                X_sfh_in_situ = np.array(
                    [sfh_data_igal[j] for j in range(n_met, 2 * n_met)]
                )

                delta_mstar_in_situ_table[igal, :] = np.sum(X_sfh_in_situ, axis=0)
                delta_mstar_tot_table[igal, :] = np.sum(X_sfh_tot, axis=0)

    return delta_mstar_tot_table, delta_mstar_in_situ_table, tarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["disk", "bulge"])
    parser.add_argument("-indrn", default=DRN_LCRC_APH2)
    parser.add_argument("-bname", default=BNAME_APH2)
    parser.add_argument("-outdrn", default=DRN_LCRC_APH2)

    parser.add_argument("-istart", default=0, type=int)
    parser.add_argument("-iend", default=-1, type=int)

    args = parser.parse_args()
    component = args.component
    indrn = args.indrn
    bname = args.bname
    outdrn = args.outdrn
    istart = args.istart
    iend = args.iend

    if iend == -1:
        iend = None

    fn = os.path.join(indrn, bname)

    if component == "disk":
        colname = DISK_COLNAME
    else:
        colname = BULGE_COLNAME

    start = time()
    dmstar_tot, dmstar_in_situ, tarr = extract_galacticus_sfh_tables(
        fn, colname, istart=istart, iend=iend
    )
    end = time()
    runtime = end - start

    msg = "Runtime to extract {0} SFH for {1} galaxies = {2:.2f} seconds"
    print(msg.format(component, dmstar_tot.shape[0], runtime))

    np.save(os.path.join(outdrn, "delta_mstar_tot_{}".format(component)), dmstar_tot)

    np.save(
        os.path.join(outdrn, "delta_mstar_in_situ_{}".format(component)), dmstar_in_situ
    )

    np.save(os.path.join(outdrn, "tarr_{}".format(component)), tarr)
