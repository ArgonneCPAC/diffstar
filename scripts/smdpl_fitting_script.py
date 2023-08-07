"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import argparse
import os
import subprocess
from time import time

import h5py
import numpy as np
from diffmah.fit_mah_helpers import get_loss_data as get_diffmah_loss_data
from diffmah.fit_mah_helpers import log_mah_mse_loss_and_grads
from diffmah.utils import jax_adam_wrapper as diffmah_fitter
from mpi4py import MPI

from diffstar.fit_smah_helpers import (
    SSFRH_FLOOR,
    get_header,
    get_loss_data_default,
    get_outline_default,
    loss_default,
    loss_grad_default_np,
)
from diffstar.utils import minimizer_wrapper

TMP_OUTPAT = "_tmp_smah_fits_rank_{0}.dat"
MIN_MASS_CUT = 7.0
FSTAR_TDELAY = 1.0


def _write_collated_data(outname, data, header):
    nrows, ncols = np.shape(data)
    colnames = header[1:].strip().split()
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

    parser.add_argument("indir", help="Input directory", default="BEBOP")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)

    parser.add_argument(
        "-mass_fit_min",
        help="Minimum mass included in stellar mass histories.",
        type=float,
        default=MIN_MASS_CUT,
    )
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir

    rank_basepat = args.outbase + TMP_OUTPAT
    rank_outname = os.path.join(outdir, rank_basepat).format(rank)

    _smah_data = load_smdpl_data(indir)
    halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data

    _smah_data = load_fit_mah()
    mah_fit_params, logmp = _smah_data

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 5
    else:
        nhalos_tot = len(halo_ids)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    halo_ids_for_rank = halo_ids[indx]
    log_smahs_for_rank = log_smahs[indx]
    sfrhs_for_rank = sfrhs[indx]
    mah_params_for_rank = mah_fit_params[indx]
    logmp_for_rank = logmp[indx]
    nhalos_for_rank = len(halo_ids_for_rank)

    get_loss_data = get_loss_data_default
    loss_func = loss_default
    loss_func_deriv = loss_grad_default_np
    get_outline = get_outline_default
    header = get_header
    kwargs = {
        "fstar_tdelay": FSTAR_TDELAY,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": SSFRH_FLOOR,
    }

    start = time()
    header = header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgsmah = log_smahs_for_rank[i, :]
            sfrh = sfrhs_for_rank[i, :]
            logmp_halo = logmp_for_rank[i]

            dmah_p_init, dmah_loss_data = get_diffmah_loss_data(tarr, lgmah, lgm_min)
            _res = diffmah_fitter(
                log_mah_mse_loss_and_grads,
                dmah_p_init,
                dmah_loss_data,
                diffmah_nstep,
                n_warmup=1,
            )
            dmah_p_best, dmah_loss_best, __, __, dmah_fit_terminates = _res

            p_init, loss_data = get_loss_data(
                tarr, dt, sfrh, lgsmah, logmp_halo, mah_params, **kwargs
            )
            _res = minimizer_wrapper(loss_func, loss_func_deriv, p_init, loss_data)
            p_best, loss_best, success = _res
            outline = get_outline(halo_id, loss_data, p_best, loss_best, success)

            fout.write(outline)

    comm.Barrier()
    end = time()

    msg = (
        "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
    )
    if rank == 0:
        runtime = end - start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        data_collection = [np.loadtxt(fn) for fn in fit_data_fnames]
        all_fit_data = np.concatenate(data_collection)
        outname = os.path.join(args.outdir, args.outbase)
        outname = outname + ".h5"
        _write_collated_data(outname, all_fit_data, header)

        #  clean up temporary files
        _remove_basename = pat.replace("{0}", "*")
        command = "rm -rf " + _remove_basename
        raw_result = subprocess.check_output(command, shell=True)
