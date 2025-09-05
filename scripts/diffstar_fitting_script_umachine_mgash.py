import argparse
import os
import subprocess
from glob import glob
from time import time

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams
from mpi4py import MPI

import diffstar.fitting_helpers.diffstar_fitting_helpers_mgash as dfh
from diffstar.data_loaders.load_smah_data import (
    FB_SMDPL,
    T0_SMDPL,
    load_smdpl_diffmah_fits,
    load_SMDPL_DR1_data,
    load_SMDPL_nomerging_data,
)

LOGMP0_MIN = 10.5
MIN_MASS_CUT = 7.0

SMDPL_NOMERGING_SFH_LCRC_DRN = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/"
SMDPL_DR1_SFH_LCRC_DRN = (
    "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/a_1.000000/"
)

SMDPL_NOMERGING_DIFFMAH_LCRC_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffmah_tpeak_fits/"
)
SMDPL_DR1_DIFFMAH_LCRC_DRN = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/diffmah_tpeak_fits/"

TMP_OUTPAT = "tmp_sfh_fits_rank_{0}.dat"
NUM_SUBVOLS_SMDPL = 576


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "sim_name", help="Simulation name", choices=["DR1", "DR1_nomerging"]
    )
    # parser.add_argument("-outbase", help="Basename of the output hdf5 file")
    # parser.add_argument(
    #     "-indir_diffmah", help="Directory of mah parameters", default=BEBOP_SMDPL_MAH
    # )
    # parser.add_argument("-indir_sfh", help="Input directory", default=BEBOP_SMDPL)
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
        default=MIN_MASS_CUT,
    )
    parser.add_argument(
        "-ssfrh_floor",
        help="Clipping floor for sSFH",
        type=float,
        default=dfh.SSFRH_FLOOR,
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument("-iend", help="Last subvolume in loop", type=int, default=-1)
    parser.add_argument(
        "-num_subvols_tot", help="Total # subvols", type=int, default=NUM_SUBVOLS_SMDPL
    )

    parser.add_argument(
        "-logmp0_min",
        help="Minimum mass required to run the diffstar fitter",
        type=float,
        default=LOGMP0_MIN,
    )

    args = parser.parse_args()

    # outbase = args.outbase
    istart, iend = args.istart, args.iend
    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    logmp0_min = args.logmp0_min

    HEADER, colnames_out = dfh.get_header()

    kwargs = {
        "fstar_tdelay": args.fstar_tdelay,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": args.ssfrh_floor,
    }

    start = time()

    if args.sim_name == "DR1":
        indir_sfh = SMDPL_DR1_SFH_LCRC_DRN
        indir_diffmah = SMDPL_DR1_DIFFMAH_LCRC_DRN
        subvol_diffmah_pat = "diffmah_fits_subvol_{}.hdf5"
    elif args.sim_name == "DR1_nomerging":
        indir_sfh = SMDPL_NOMERGING_SFH_LCRC_DRN
        indir_diffmah = SMDPL_NOMERGING_DIFFMAH_LCRC_DRN
        subvol_diffmah_pat = "subvol_{}_diffmah_fits.h5"
    else:
        raise NotImplementedError

    all_avail_subvol_names = [
        os.path.basename(drn) for drn in glob(os.path.join(indir_sfh, "subvol_*"))
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
        comm.Barrier()
        isubvol_start = time()

        nchar_subvol = len(str(num_subvols_tot))
        subvol_str = f"subvol_{isubvol:0{nchar_subvol}d}"

        subvolumes_i = [isubvol]

        if args.sim_name == "DR1":
            _sfh_data = load_SMDPL_DR1_data(subvolumes_i, indir_sfh)
        elif args.sim_name == "DR1_nomerging":
            _sfh_data = load_SMDPL_nomerging_data(subvolumes_i, indir_sfh)
        else:
            raise NotImplementedError
        halo_ids, log_smahs_sim, sfhs_sim, t_smdpl = _sfh_data[:4]

        subvol_diffmah_str = subvol_diffmah_pat.format(isubvol)
        _mah_fits = load_smdpl_diffmah_fits(subvol_diffmah_str, data_drn=indir_diffmah)
        mah_params, logmp0, diffmah_loss, n_points_per_diffmah_fit = _mah_fits

        msg = "Mismatch: Nrows(sfh_data)={0} Nrows(diffmah_data)={1}"
        n_diffstar = halo_ids.size
        n_diffmah = logmp0.size
        assert n_diffstar == n_diffmah, msg.format(n_diffstar, n_diffmah)

        has_diffmah_fit = (diffmah_loss > 0) & (logmp0 > logmp0_min)

        if rank == 0:
            print("Number of galaxies in subvolume = {}".format(len(halo_ids)))

        # Get data for rank
        if args.test:
            nhalos_tot = nranks * 5
        else:
            nhalos_tot = len(halo_ids)

        indx_all = np.arange(0, nhalos_tot).astype("i8")
        indx = np.array_split(indx_all, nranks)[rank]

        halo_ids_for_rank = halo_ids[indx]
        log_smahs_for_rank = log_smahs_sim[indx]
        sfh_sim_for_rank = sfhs_sim[indx]
        mah_params_for_rank = mah_params._make([x[indx] for x in mah_params])
        logmp0_for_rank = logmp0[indx]
        has_diffmah_fit_for_rank = has_diffmah_fit[indx]

        nhalos_for_rank = len(halo_ids_for_rank)

        rank_basepat = "_".join((subvol_str, TMP_OUTPAT))
        rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

        comm.Barrier()
        with open(rank_outname, "w") as fout:
            fout.write(HEADER)

            for i in range(nhalos_for_rank):
                halo_id = halo_ids_for_rank[i]
                lgsmah = log_smahs_for_rank[i, :]
                sfh = sfh_sim_for_rank[i, :]
                mah_params = DiffmahParams(*[x[i] for x in mah_params_for_rank])
                logmp0_halo = logmp0_for_rank[i]
                halo_has_diffmah_fit = has_diffmah_fit_for_rank[i]

                run_fitter = (logmp0_halo > logmp0_min) & halo_has_diffmah_fit
                if run_fitter:
                    _res = dfh.diffstar_fitter(
                        t_smdpl,
                        sfh,
                        mah_params,
                        lgt0=np.log10(T0_SMDPL),
                        fb=FB_SMDPL,
                    )
                    sfh_params_best, diffstar_loss, diffstar_fit_success = _res
                    outline = dfh.get_outline(
                        halo_id, sfh_params_best, diffstar_loss, diffstar_fit_success
                    )
                else:
                    outline = dfh.get_outline_nofit(halo_id)

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
            outbn = f"diffstar_fits_subvol_{isubvol}.hdf5"
            outfn = os.path.join(args.outdir, outbn)

            # fitsmah.write_collated_data(outfn, subvol_i_fit_results, chunk_arr=None)
            dfh.write_collated_data(outfn, subvol_i_fit_results, colnames_out)

            # clean up ASCII data for subvol_i
            bnpat = subvol_str + "*.dat"
            fnpat = os.path.join(args.outdir, bnpat)
            command = "rm " + fnpat
            subprocess.os.system(command)
