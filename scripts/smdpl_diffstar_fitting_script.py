"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""
import argparse
import os
import subprocess
from glob import glob
from time import time

import h5py
import numpy as np
from diffstar.fit_smah_helpers import (
    SSFRH_FLOOR,
    get_header,
    get_loss_data_default,
    get_outline_default,
    loss_default,
    loss_grad_default_np,
)
from diffstar.utils import minimizer_wrapper
from mpi4py import MPI

N_SUBVOL_SMDPL = 576
from load_smdpl import LOGSM_MIN, load_diffmah_fit_params, load_smdpl_histories

TMP_OUTPAT = "_tmp_mah_fits_rank_{0}.dat"
TODAY = 13.8

TASSO = os.path.join(
    "/Users/aphearin/work/DATA/SMDPL",
    "dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000",
)
TASSO_DIFFMAH = "/Users/aphearin/work/DATA/SMDPL/diffmah_fits"
BEBOP = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/"
BEBOP_DIFFMAH = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/diffmah_fits"


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = get_header()[1:].strip().split()
    assert len(colnames) == ncols, "data mismatched with header"
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Input directory")
    parser.add_argument("diffmah_indir", help="Input directory of diffmah fits")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument("-nstep", help="Num opt steps per halo", type=int, default=200)
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument(
        "-iend", help="First subvolume in loop", type=int, default=N_SUBVOL_SMDPL
    )

    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    nstep = args.nstep
    istart, iend = args.istart, args.iend

    if args.indir == "TASSO":
        indir = TASSO
    elif args.indir == "BEBOP":
        indir = BEBOP
    else:
        indir = args.indir

    if args.diffmah_indir == "TASSO":
        diffmah_indir = TASSO_DIFFMAH
    elif args.diffmah_indir == "BEBOP":
        diffmah_indir = BEBOP_DIFFMAH
    else:
        diffmah_indir = args.diffmah_indir

    all_avail_subvol_names = [
        os.path.basename(drn) for drn in glob(os.path.join(indir, "subvol_*"))
    ]
    all_avail_subvolumes = [int(s.split("_")[1]) for s in all_avail_subvol_names]
    if args.test:
        subvolumes = [
            all_avail_subvolumes[0],
        ]
    else:
        subvolumes = np.arange(istart, iend)

    fitter_kwargs = {
        "fstar_tdelay": 1.0,
        "mass_fit_min": LOGSM_MIN,
        "ssfrh_floor": SSFRH_FLOOR,
    }

    ngals_complete_fits = 0
    for isubvol in subvolumes:
        isubvol_start = time()

        subvolumes_i = [isubvol]
        mock, tarr, dt, lgmh_min = load_smdpl_histories(indir, subvolumes_i)
        if rank == 0:
            print("Number of galaxies in mock = {}".format(len(mock)))

        _res = load_diffmah_fit_params(diffmah_indir, subvolumes_i)
        diffmah_fit_data_isubvol, logmp_isubvol = _res

        # Get data for rank
        if args.test:
            nhalos_tot = nranks * 5
        else:
            nhalos_tot = len(mock["halo_id"])
        _a = np.arange(0, nhalos_tot).astype("i8")
        indx = np.array_split(_a, nranks)[rank]

        halo_ids_for_rank = mock["halo_id"][indx]
        nhalos_for_rank = len(halo_ids_for_rank)

        log_mahs_for_rank = mock["log_mah"][indx]
        logmp_for_rank = logmp_isubvol[indx]
        mah_params_for_rank = diffmah_fit_data_isubvol[indx]

        log_smahs_for_rank = mock["log_smahs"][indx]
        sfrhs_for_rank = mock["sfr_history_main_prog"][indx]

        subvol_string = "subvol_{0}".format(isubvol)
        rank_basepat = subvol_string + "_" + args.outbase + TMP_OUTPAT
        rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

        header = get_header()
        with open(rank_outname, "w") as fout:
            fout.write(header)

            for i in range(nhalos_for_rank):
                halo_id = halo_ids_for_rank[i]
                lgmah = log_mahs_for_rank[i, :]
                sfrh = sfrhs_for_rank[i, :]
                lgsmah = log_smahs_for_rank[i, :]
                logmp_halo = lgmah[-1]
                mah_params = mah_params_for_rank[i, :]

                p_init, loss_data = get_loss_data_default(
                    tarr, dt, sfrh, lgsmah, logmp_halo, mah_params, **fitter_kwargs
                )
                _res = minimizer_wrapper(
                    loss_default, loss_grad_default_np, p_init, loss_data
                )
                p_best, loss_best, success = _res
                outline = get_outline_default(
                    halo_id, loss_data, p_best, loss_best, success
                )

                fout.write(outline)

        comm.Barrier()
        isubvol_end = time()
        ngals_complete_fits += len(mock)

        msg = "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
        if rank == 0:
            print("\nFinished with subvolume {}".format(isubvol))
            runtime = isubvol_end - isubvol_start
            print(msg.format(nhalos_tot, nranks, runtime))

            #  collate data from ranks and rewrite to disk
            pat = os.path.join(args.outdir, rank_basepat)
            fit_data_fnames = [pat.format(i) for i in range(nranks)]
            data_collection = [np.loadtxt(fn) for fn in fit_data_fnames]
            all_fit_data = np.concatenate(data_collection)
            outdrn = os.path.join(args.outdir, subvol_string)
            os.makedirs(outdrn, exist_ok=True)
            outname = os.path.join(outdrn, args.outbase)
            _write_collated_data(outname, all_fit_data)

            #  clean up temporary files
            _remove_basename = pat.replace("{0}", "*")
            command = "rm -rf " + _remove_basename
            raw_result = subprocess.check_output(command, shell=True)

    end = time()
    final_runtime = end - start
    msg = "Final runtime to fit {0} galaxies with {1} ranks = {2:.2f} hours\n\n"
    if rank == 0:
        print(msg.format(ngals_complete_fits, nranks, final_runtime / 3600))
