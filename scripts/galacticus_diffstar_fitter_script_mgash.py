import argparse
import os
import subprocess
from time import time

import h5py
import numpy as np
from diffmah.diffmah_kernels import DiffmahParams
from mpi4py import MPI

import diffstar.fitting_helpers.diffstar_fitting_helpers as dfh
from diffstar.data_loaders import load_galacticus_sfh as lgs
from diffstar.data_loaders import load_precomputed_diffmah_fits
from diffstar.utils import cumulative_mstar_formed

DRN_POBOY = "/Users/aphearin/work/DATA/Galacticus/diffstarpop_data"
DRN_LCRC = "/lcrc/project/halotools/Galacticus/diffstarpop_data"
BNAME_APH2 = "galacticus_11To14.2Mhalo_SFHinsitu_AHearin.hdf5"

LOGMP0_MIN = 10.5
MIN_MASS_CUT = 7.0
MIN_LOGSM_Z0 = 8.0

NHALOS_TEST = 30

TMP_OUTPAT = "tmp_sfh_fits_rank_{0}_{1}.dat"


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("sfh_type", choices=["in_situ", "in_plus_ex_situ"])

    parser.add_argument("-indir", help="Input directory", default=DRN_LCRC)
    parser.add_argument("-inbn", help="Input basename", default=lgs.BNAME_SFH_DATA)
    parser.add_argument("-outdir", help="Output directory", default="")
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
    parser.add_argument(
        "-logmp0_min",
        help="Minimum mass required to run the diffstar fitter",
        type=float,
        default=LOGMP0_MIN,
    )

    args = parser.parse_args()
    sfh_type = args.sfh_type
    indir = args.indir
    inbn = args.inbn
    outdir = args.outdir
    logmp0_min = args.logmp0_min
    is_test = args.test

    fn_sfh_block = os.path.join(indir, inbn)
    fn_diffmah = os.path.join(indir, "diffmah_fits.h5")

    if outdir == "":
        outdir = indir

    HEADER, colnames_out = dfh.get_header()

    kwargs = {
        "fstar_tdelay": args.fstar_tdelay,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": args.ssfrh_floor,
    }

    comm.Barrier()

    if sfh_type == "in_situ":
        sfh_colname = "sfh_in_situ"
    elif sfh_type == "in_plus_ex_situ":
        sfh_colname = "sfh_tot"

    with h5py.File(fn_sfh_block, "r") as hdf:
        nhalos_tot = hdf["sfh_in_situ"].shape[0]

    _a = np.arange(0, nhalos_tot).astype("i8")
    indx_for_rank = np.array_split(_a, nranks)[rank]
    istart = indx_for_rank[0]
    iend = indx_for_rank[-1] + 1

    sfh_data_for_rank = lgs.load_galacticus_sfh_data_block(fn_sfh_block, istart, iend)
    tarr = sfh_data_for_rank["tarr"]
    T0 = tarr[-1]

    _res = load_precomputed_diffmah_fits(fn_diffmah, T0, istart=istart, iend=iend)
    mah_params_for_rank, logmp0_for_rank = _res[:2]
    diffmah_loss_for_rank, n_points_per_diffmah_fit_for_rank = _res[2:]

    indices_for_rank = np.arange(nhalos_tot).astype(int)[istart:iend]
    halo_ids_for_rank = np.copy(indices_for_rank)
    nhalos_for_rank = halo_ids_for_rank.size
    if is_test:
        nhalos_for_rank = NHALOS_TEST

    has_diffmah_fit_for_rank = diffmah_loss_for_rank > 0
    has_diffmah_fit_for_rank &= logmp0_for_rank > logmp0_min

    if rank == 0:
        print("Number of galaxies for rank 0 = {}".format(nhalos_for_rank))

    rank_outname = os.path.join(outdir, TMP_OUTPAT).format(rank, sfh_type)

    comm.Barrier()

    fitting_start = time()
    with open(rank_outname, "w") as fout:
        fout.write(HEADER)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]

            mah_params = DiffmahParams(*[x[i] for x in mah_params_for_rank])
            logmp0_halo = logmp0_for_rank[i]
            halo_has_diffmah_fit = has_diffmah_fit_for_rank[i]

            sfh = sfh_data_for_rank[sfh_colname][i]
            logsm_z0 = np.log10(cumulative_mstar_formed(tarr, sfh)[-1])

            run_fitter = logmp0_halo > logmp0_min
            run_fitter &= halo_has_diffmah_fit
            run_fitter &= logsm_z0 > MIN_LOGSM_Z0

            if run_fitter:
                _res = dfh.diffstar_fitter(
                    tarr, sfh, mah_params, lgt0=np.log10(T0), fb=lgs.FB
                )
                sfh_params_best, diffstar_loss, diffstar_fit_success = _res
                outline = dfh.get_outline(
                    halo_id, sfh_params_best, diffstar_loss, diffstar_fit_success
                )
            else:
                outline = dfh.get_outline_nofit(halo_id)

            fout.write(outline)

    comm.Barrier()
    fitting_end = time()

    msg = (
        "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
    )
    if rank == 0:
        runtime = fitting_end - fitting_start
        print(msg.format(nhalos_for_rank * nranks, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(outdir, TMP_OUTPAT)
        fit_data_fnames = [pat.format(i, sfh_type) for i in range(nranks)]
        collector = []
        for fit_fn in fit_data_fnames:
            assert os.path.isfile(fit_fn), fit_fn
            fit_data = np.genfromtxt(fit_fn, dtype="str")
            collector.append(fit_data)
        subvol_i_fit_results = np.concatenate(collector)

        outbn = f"diffstar_fits_{sfh_type}.hdf5"
        outfn = os.path.join(outdir, outbn)

        # fitsmah.write_collated_data(outfn, subvol_i_fit_results, chunk_arr=None)
        dfh.write_collated_data(outfn, subvol_i_fit_results, colnames_out)

        # clean up ASCII data for subvol_i
        for fn in fit_data_fnames:
            command = "rm " + fn
            subprocess.os.system(command)
