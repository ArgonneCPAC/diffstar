import argparse
import os
import subprocess
from glob import glob
from time import time

import numpy as np
from mpi4py import MPI

import diffstar.fitting_helpers.fit_smah_helpers_tpeak as fitsmah
from diffstar.data_loaders.load_smah_data import load_fit_mah_tpeak, load_SMDPL_data
from diffstar.fitting_helpers.utils import minimizer_wrapper

BEBOP_SMDPL = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/"
BEBOP_SMDPL_MAH = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffmah_tpeak_fits/"
)

TMP_OUTPAT = "tmp_sfh_fits_rank_{0}.dat"
NUM_SUBVOLS_SMDPL = 576


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-outbase", help="Basename of the output hdf5 file")
    parser.add_argument(
        "-indir_diffmah", help="Directory of mah parameters", default=BEBOP_SMDPL_MAH
    )
    parser.add_argument("-indir", help="Input directory", default=BEBOP_SMDPL)
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
        default=fitsmah.MIN_MASS_CUT,
    )
    parser.add_argument(
        "-ssfrh_floor",
        help="Clipping floor for sSFH",
        type=float,
        default=fitsmah.SSFRH_FLOOR,
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument("-iend", help="Last subvolume in loop", type=int, default=-1)
    parser.add_argument(
        "-num_subvols_tot", help="Total # subvols", type=int, default=NUM_SUBVOLS_SMDPL
    )

    args = parser.parse_args()

    indir = args.indir
    indir_diffmah = args.indir_diffmah
    outbase = args.outbase
    istart, iend = args.istart, args.iend
    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    HEADER, colnames_out = fitsmah.get_header()

    kwargs = {
        "fstar_tdelay": args.fstar_tdelay,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": args.ssfrh_floor,
    }

    start = time()

    all_avail_subvol_names = [
        os.path.basename(drn) for drn in glob(os.path.join(indir, "subvol_*"))
    ]
    all_avail_subvolumes = [int(s.split("_")[1]) for s in all_avail_subvol_names]
    all_avail_subvolumes = sorted(all_avail_subvolumes)

    if args.test:
        subvolumes = [
            all_avail_subvolumes[0],
        ]
    else:
        subvolumes = np.arange(istart, iend)

    for isubvol in subvolumes:
        isubvol_start = time()

        nchar_subvol = len(str(num_subvols_tot))
        subvol_str = f"subvol_{isubvol:0{nchar_subvol}d}"

        subvolumes_i = [isubvol]

        subvol_data_str = indir
        _data = load_SMDPL_data(subvolumes_i, subvol_data_str)
        halo_ids, log_smahs, sfrhs, tarr, dt, log_mahs, logmp = _data

        subvol_diffmah_str = f"{subvol_str}_diffmah_fits.h5"
        mah_fit_params, logmp_fit, t_peak_arr = load_fit_mah_tpeak(
            subvol_diffmah_str, data_drn=indir_diffmah
        )

        if rank == 0:
            print("Number of galaxies in mock = {}".format(len(halo_ids)))

        # Get data for rank
        if args.test:
            nhalos_tot = nranks * 5
        else:
            nhalos_tot = len(halo_ids)

        indx_all = np.arange(0, nhalos_tot).astype("i8")
        indx = np.array_split(indx_all, nranks)[rank]

        halo_ids_for_rank = halo_ids[indx]
        log_smahs_for_rank = log_smahs[indx]
        sfrhs_for_rank = sfrhs[indx]
        mah_params_for_rank = mah_fit_params[indx]
        logmp_for_rank = logmp[indx]
        t_peak_for_rank = t_peak_arr[indx]

        nhalos_for_rank = len(halo_ids_for_rank)

        rank_basepat = "_".join((subvol_str, outbase, TMP_OUTPAT))
        rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

        # breakpoint()
        with open(rank_outname, "w") as fout:
            fout.write(HEADER)

            for i in range(nhalos_for_rank):
                halo_id = halo_ids_for_rank[i]
                lgsmah = log_smahs_for_rank[i, :]
                sfrh = sfrhs_for_rank[i, :]
                mah_params = mah_params_for_rank[i]
                logmp_halo = logmp_for_rank[i]
                t_peak = t_peak_for_rank[i]

                p_init, loss_data = fitsmah.get_loss_data_default(
                    tarr, dt, sfrh, lgsmah, logmp_halo, mah_params, t_peak, **kwargs
                )
                _res = minimizer_wrapper(
                    fitsmah.loss_default_clipssfrh,
                    fitsmah.loss_grad_default_clipssfrh_np,
                    p_init,
                    loss_data,
                )
                p_best, loss_best, success = _res
                outline = fitsmah.get_outline_default(
                    halo_id, loss_data, p_best, loss_best, success
                )

                fout.write(outline)

        comm.Barrier()
        isubvol_end = time()

        msg = "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
        if rank == 0:
            print("\nFinished with subvolume {}".format(isubvol))
            runtime = isubvol_end - isubvol_start
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

            subvol_str = f"subvol_{isubvol:0{nchar_subvol}d}"
            outbn = "_".join((subvol_str, outbase)) + ".h5"
            outfn = os.path.join(args.outdir, outbn)

            # fitsmah.write_collated_data(outfn, subvol_i_fit_results, chunk_arr=None)
            fitsmah.write_collated_data(outfn, subvol_i_fit_results, colnames_out)

            # clean up ASCII data for subvol_i
            bnpat = subvol_str + "*.dat"
            fnpat = os.path.join(args.outdir, bnpat)
            command = "rm " + fnpat
            subprocess.os.system(command)
