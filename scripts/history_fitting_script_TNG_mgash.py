import argparse
import os
import subprocess
from time import time

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams
from mpi4py import MPI

import diffstar.fitting_helpers.diffstar_fitting_helpers_mgash as dfh
from diffstar.data_loaders.load_smah_data import (
    load_tng_data,
    FB_TNG,
)
from diffstar.data_loaders import load_precomputed_diffmah_fits
from diffstar.fitting_helpers.utils import minimizer_wrapper

BEBOP_TNG = "/lcrc/project/halotools/alarcon/data/"
BEBOP_TNG_MAH = "/lcrc/project/halotools/alarcon/results/tng_diffmah_tpeak/"

TMP_OUTPAT = "tmp_sfh_fits_rank_{0}.dat"
NCHUNKS = 20


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-outbase", help="Basename of the output hdf5 file")
    parser.add_argument(
        "-indir_diffmah", help="Directory of mah parameters", default=BEBOP_TNG_MAH
    )
    parser.add_argument("-indir", help="Input directory", default=BEBOP_TNG)
    parser.add_argument(
        "-fstar_tdelay",
        help="Time interval in Gyr for fstar definition.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-mass_fit_min",
        help="Minimum mass included in stellar mass histories.",
        type=float,
        default=dfh.MIN_MASS_CUT,
    )
    parser.add_argument(
        "-ssfrh_floor",
        help="Clipping floor for sSFH",
        type=float,
        default=dfh.SSFRH_FLOOR,
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir
    indir_diffmah = args.indir_diffmah
    outbase = args.outbase
    nchunks = args.nchunks
    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    HEADER, colnames_out = dfh.get_header()

    kwargs = {
        "fstar_tdelay": args.fstar_tdelay,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": args.ssfrh_floor,
    }

    start = time()

    _data = load_tng_data(indir)
    halo_ids, log_smahs, sfrhs, tarr, dt, log_mahs, logmp = _data
    T0 = tarr[-1]

    if rank == 0:
        print("Number of galaxies in mock = {}".format(len(halo_ids)))

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 5
    else:
        nhalos_tot = len(halo_ids)

    _a = np.arange(0, nhalos_tot).astype("i8")
    indx_for_rank = np.array_split(_a, nranks)[rank]
    istart = indx_for_rank[0]
    iend = indx_for_rank[-1] + 1

    _res = load_precomputed_diffmah_fits(indir_diffmah, T0, istart=istart, iend=iend)
    mah_params_for_rank, logmp_for_rank = _res[:2]

    halo_ids_for_rank = halo_ids[indx_for_rank]
    log_smahs_for_rank = log_smahs[indx_for_rank]
    sfrhs_for_rank = sfrhs[indx_for_rank]

    nhalos_for_rank = len(halo_ids_for_rank)

    ichunk_start = time()

    nhalos_tot = comm.reduce(nhalos_for_rank, op=MPI.SUM)

    rank_basepat = TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

    # If final collated chunk filename already exists,
    # tell every rank to skip to next chunk.
    bname = os.path.basename(rank_outname)
    outbn = "_".join(bname.split("_")[:4]) + ".hdf5"
    outfn = os.path.join(outdir, outbn)

    with open(rank_outname, "w") as fout:
        fout.write(HEADER)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgsmah = log_smahs_for_rank[i, :]
            sfrh = sfrhs_for_rank[i, :]
            mah_params = DiffmahParams(*mah_params_for_rank[i])
            logmp_halo = logmp_for_rank[i]

            _res = dfh.diffstar_fitter(
                tarr, sfrh, mah_params, lgt0=np.log10(T0), fb=FB_TNG
            )
            sfh_params_best, diffstar_loss, diffstar_fit_success = _res
            outline = dfh.get_outline(
                halo_id, sfh_params_best, diffstar_loss, diffstar_fit_success
            )

            fout.write(outline)

    ichunk_end = time()
    comm.Barrier()

    msg = "\n\nWallclock runtime to fit {0} halos with {1} ranks = {2:.1f} seconds\n\n"
    if rank == 0:
        runtime = ichunk_end - ichunk_start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        collector = []
        for fit_fn in fit_data_fnames:
            assert os.path.isfile(fit_fn)
            fit_data = np.genfromtxt(fit_fn, dtype="str")
            collector.append(fit_data)
        subvol_i_fit_results = np.concatenate(collector)

        fit_data_bnames = [os.path.basename(fn) for fn in fit_data_fnames]
        outbn = "diffstar_tng_fits.hdf5"
        outfn = os.path.join(outdir, outbn)

        # fitsmah.write_collated_data(outfn, chunk_fit_results, colnames_out)
        dfh.write_collated_data(outfn, subvol_i_fit_results, colnames_out)

        # clean up ASCII data for subvol_i
        bn = fit_data_bnames[0]
        bnpat = "_".join(bn.split("_")[:-1]) + "_*.dat"
        fnpat = os.path.join(outdir, bnpat)
        command = "rm " + fnpat
        subprocess.os.system(command)
