"""Script to fit Bolshoi or Multidark MAHs with a smooth model."""
import argparse
import os
import subprocess
from time import time

import h5py
import numpy as np
from mpi4py import MPI

from diffstar.data_loaders.load_smah_data import (
    BEBOP,
    TASSO,
    load_bolshoi_data,
    load_bolshoi_small_data,
    load_fit_mah,
    load_mdpl_data,
    load_mdpl_small_data,
    load_tng_data,
    load_tng_small_data,
)
from diffstar.fitting_helpers.fit_smah_helpers import (
    MIN_MASS_CUT,
    SSFRH_FLOOR,
    get_header,
    get_loss_data_default,
    get_loss_data_fixed_depl_noquench,
    get_loss_data_fixed_hi,
    get_loss_data_fixed_hi_depl,
    get_loss_data_fixed_hi_rej,
    get_loss_data_fixed_noquench,
    get_loss_data_free,
    get_outline_default,
    get_outline_fixed_depl_noquench,
    get_outline_fixed_hi,
    get_outline_fixed_hi_depl,
    get_outline_fixed_hi_rej,
    get_outline_fixed_noquench,
    get_outline_free,
    loss_default,
    loss_fixed_depl_noquench,
    loss_fixed_depl_noquench_deriv_np,
    loss_fixed_hi,
    loss_fixed_hi_depl,
    loss_fixed_hi_depl_deriv_np,
    loss_fixed_hi_deriv_np,
    loss_fixed_hi_rej,
    loss_fixed_hi_rej_deriv_np,
    loss_fixed_noquench,
    loss_fixed_noquench_deriv_np,
    loss_free,
    loss_free_deriv_np,
    loss_grad_default_np,
)
from diffstar.utils import minimizer_wrapper

TMP_OUTPAT = "_tmp_smah_fits_rank_{0}.dat"
TODAY = 13.8


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

    parser.add_argument(
        "simulation", help="name of the simulation (used to select the data loader)"
    )
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument(
        "modelname",
        help="Version of the model and loss",
        choices=(
            "default",
            "free",
            "fixed_noquench",
            "fixed_hi",
            "fixed_hi_rej",
            "fixed_hi_depl",
            "fixed_depl_noquench",
        ),
        default="default",
    )
    parser.add_argument("-indir", help="Input directory", default="BEBOP")
    parser.add_argument("-fitmahfn", help="Filename of fit mah parameters")
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument(
        "-gal_type",
        help="Galaxy type (only relevant for Bolshoi and MDPl2)",
        default="cens",
    )
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
        default=SSFRH_FLOOR,
    )
    args = parser.parse_args()

    start = time()

    args = parser.parse_args()
    rank_basepat = args.outbase + TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

    if args.indir == "TASSO":
        indir = TASSO
    elif args.indir == "BEBOP":
        indir = BEBOP
    else:
        indir = args.indir

    if args.simulation == "bpl":
        _smah_data = load_bolshoi_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    elif args.simulation == "tng":
        _smah_data = load_tng_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    elif args.simulation == "mdpl":
        _smah_data = load_mdpl_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    elif args.simulation == "bpl_small":
        _smah_data = load_bolshoi_small_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    elif args.simulation == "tng_small":
        _smah_data = load_tng_small_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    elif args.simulation == "mdpl_small":
        _smah_data = load_mdpl_small_data(args.gal_type, data_drn=indir)
        halo_ids, log_smahs, sfrhs, tarr, dt = _smah_data
        _smah_data = load_fit_mah(args.fitmahfn, data_drn=indir)
        mah_fit_params, logmp = _smah_data
    else:
        raise NotImplementedError

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

    if args.modelname == "default":
        get_loss_data = get_loss_data_default
        loss_func = loss_default
        loss_func_deriv = loss_grad_default_np
        get_outline = get_outline_default
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "free":
        get_loss_data = get_loss_data_free
        loss_func = loss_free
        loss_func_deriv = loss_free_deriv_np
        get_outline = get_outline_free
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "fixed_noquench":
        get_loss_data = get_loss_data_fixed_noquench
        loss_func = loss_fixed_noquench
        loss_func_deriv = loss_fixed_noquench_deriv_np
        get_outline = get_outline_fixed_noquench
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "fixed_hi":
        get_loss_data = get_loss_data_fixed_hi
        loss_func = loss_fixed_hi
        loss_func_deriv = loss_fixed_hi_deriv_np
        get_outline = get_outline_fixed_hi
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "fixed_hi_rej":
        get_loss_data = get_loss_data_fixed_hi_rej
        loss_func = loss_fixed_hi_rej
        loss_func_deriv = loss_fixed_hi_rej_deriv_np
        get_outline = get_outline_fixed_hi_rej
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "fixed_hi_depl":
        get_loss_data = get_loss_data_fixed_hi_depl
        loss_func = loss_fixed_hi_depl
        loss_func_deriv = loss_fixed_hi_depl_deriv_np
        get_outline = get_outline_fixed_hi_depl
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }
    elif args.modelname == "fixed_depl_noquench":
        get_loss_data = get_loss_data_fixed_depl_noquench
        loss_func = loss_fixed_depl_noquench
        loss_func_deriv = loss_fixed_depl_noquench_deriv_np
        get_outline = get_outline_fixed_depl_noquench
        header = get_header
        kwargs = {
            "fstar_tdelay": args.fstar_tdelay,
            "mass_fit_min": args.mass_fit_min,
            "ssfrh_floor": args.ssfrh_floor,
        }

    header = header()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgsmah = log_smahs_for_rank[i, :]
            sfrh = sfrhs_for_rank[i, :]
            mah_params = mah_params_for_rank[i]
            logmp_halo = logmp_for_rank[i]

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
