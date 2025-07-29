"""Script to fit Galacticus EPS MAHs with diffmah"""

import argparse
import os
import subprocess
from time import time

import numpy as np
from diffmah.fitting_helpers import diffmah_fitter_helpers as cfh
from mpi4py import MPI

from diffstar.data_loaders import extract_galacticus_sfh_data as egsd

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"
DRN_ROOT2_POBOY = "/Users/aphearin/work/DATA/Galacticus/diffstarpop_data/fine_timesteps"
SUBDRN_APH2 = "central/scratch/sweeraso/outputs/AHearin2"
DRN_POBOY_APH2 = os.path.join(DRN_ROOT2_POBOY, SUBDRN_APH2)
BNAME_APH2 = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"

DRN_LCRC = "/lcrc/project/halotools/Galacticus/diffstarpop_data/nstep_390"

DT_FIT_MIN = 0.5  # Minimum halo lifetime in Gyr to bother running fitter


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-indir", help="Root directory storing MAH data", default=DRN_LCRC
    )
    parser.add_argument(
        "-inbn", help="Basename of file storing MAH data", default=BNAME_APH2
    )
    parser.add_argument("-outdir", help="Output directory", default="")
    parser.add_argument(
        "-outbn", help="Basename of the output hdf5 file", default="diffmah_fits.h5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument(
        "-dt_fit_min",
        help="Minimum halo lifetime in Gyr to run fitter",
        default=DT_FIT_MIN,
        type=float,
    )

    args = parser.parse_args()
    indir = args.indir
    inbn = args.inbn
    outdir = os.path.abspath(args.outdir)
    outbn = args.outbn
    is_test = args.test
    dt_fit_min = args.dt_fit_min

    fn = os.path.join(indir, inbn)
    os.makedirs(outdir, exist_ok=True)

    start = time()

    nhalos_tot = egsd.get_nhalos_tot(fn)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx_for_rank = np.array_split(_a, nranks)[rank]
    istart = indx_for_rank[0]
    iend = indx_for_rank[-1] + 1

    _res = egsd.extract_galacticus_mah_tables(fn, istart, iend)
    mahs_for_rank, times_for_rank, is_cen_for_rank, t_infall_for_rank = _res
    nhalos_for_rank = len(mahs_for_rank)

    if is_test:
        n_halos_test = 50
        nhalos_for_rank = min(nhalos_for_rank, n_halos_test)
        nhalos_tot = nranks * nhalos_for_rank

    comm.Barrier()

    t_max_for_rank = np.array([t[-1] for t in times_for_rank]).max()
    T0 = comm.allreduce(t_max_for_rank, op=MPI.MAX)
    comm.Barrier()

    rank_basepat = TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, TMP_OUTPAT).format(rank)

    comm.Barrier()
    with open(rank_outname, "w") as fout:
        fout.write(cfh.HEADER)

        rank_fit_start = time()
        for ihalo in range(nhalos_for_rank):
            t_sim = times_for_rank[ihalo]
            mah_sim = np.maximum.accumulate(mahs_for_rank[ihalo])
            dt = t_sim[-1] - t_sim[0]
            if dt < dt_fit_min:
                force_skip_fit = True
            else:
                force_skip_fit = False

            if t_sim[-1] < T0:
                t_sim = np.append(t_sim, T0)
                mah_sim = np.append(mah_sim, mah_sim[-1])

            fit_results = cfh.diffmah_fitter(
                t_sim, mah_sim, force_skip_fit=force_skip_fit
            )

            outline = cfh.get_outline(fit_results)
            fout.write(outline)

    comm.Barrier()
    rank_fit_end = time()

    msg = "\n\nWallclock runtime to fit {0} halos with {1} ranks = {2:.1f} seconds\n\n"
    if rank == 0:
        runtime = rank_fit_end - rank_fit_start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        collector = []
        for fit_fn in fit_data_fnames:
            assert os.path.isfile(fit_fn)
            fit_data = np.genfromtxt(fit_fn, dtype="str")
            collector.append(fit_data)
        chunk_fit_results = np.concatenate(collector)

        fit_data_bnames = [os.path.basename(fn) for fn in fit_data_fnames]
        outfn = os.path.join(outdir, outbn)

        cfh.write_collated_data(outfn, chunk_fit_results)

        # clean up ASCII data for subvol_i
        bn = fit_data_bnames[0]
        bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
        fnpat = os.path.join(outdir, bnpat)
        command = "rm " + fnpat
        subprocess.os.system(command)
