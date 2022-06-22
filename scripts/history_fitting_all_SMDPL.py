import numpy as np
import os
import warnings
from umachine_pyio.load_mock import load_mock_from_binaries
from astropy.cosmology import Planck15
import h5py
from diffstar.utils import _get_dt_array
from schwimmbad import MPIPool

from mpi4py import MPI
import argparse
from time import time
import sys

from diffstar.fit_smah_helpers import MIN_MASS_CUT, SSFRH_FLOOR
from diffstar.fit_smah_helpers import (
    get_header as get_header_diffstar,
    get_loss_data_default as get_loss_data_diffstar,
    loss_default as loss_func_diffstar,
    loss_grad_default_np as loss_func_deriv_diffstar,
    get_outline_default as get_outline_diffstar,
)

from diffmah.fit_mah_helpers import (
    get_header as get_header_diffmah,
    get_outline_bad_fit as get_outline_bad_fit_diffmah,
    get_loss_data as get_loss_data_diffmah,
    log_mah_mse_loss_and_grads as mse_loss_and_grads_diffmah,
    get_outline as get_outline_diffmah,
)

from diffmah.utils import jax_adam_wrapper

from diffstar.utils import minimizer_wrapper
import subprocess


BEBOP_SMDPL = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_z0_binaries/"

H_SMDPL = 0.6777
TODAY = 13.8


def load_SMDPL_data(subvols, data_drn=BEBOP_SMDPL):
    """Load the stellar mass histories from UniverseMachine simulation
    applied to the Bolshoi-Planck (BPL) simulation.

    The loaded stellar mass data has units of Msun assuming the h = H_BPL
    from the cosmology of the underlying simulation.

    The output stellar mass data has units of Msun/h, or units of
    Mstar[h=H_BPL] using the h value of the simulation.

    H_BPL is defined at the top of the module.

    Parameters
    ----------
    gal_type : string
        Name of the galaxy type of the file being loaded. Options are
            'cens': central galaxies
            'sats': satellite galaxies
            'orphans': orphan galaxies
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    halo_ids:  ndarray of shape (n_gal, )
        IDs of the halos in the file.
    log_smahs: ndarray of shape (n_gal, n_times)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfrh: ndarray of shape (n_gal, n_times)
        Star formation rate history in units of Msun/yr assuming h=1.
    bpl_t : ndarray of shape (n_times, )
        Cosmic time of each simulated snapshot in Gyr
    dt : ndarray of shape (n_times, )
        Cosmic time steps between each simulated snapshot in Gyr
    """

    galprops = ["halo_id", "sfr_history_main_prog", "mpeak_history_main_prog"]
    halos = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

    SMDPL_a = np.load("/lcrc/project/halotools/UniverseMachine/SMDPL/scale_list.npy")
    SMDPL_z = 1.0 / SMDPL_a - 1.0
    SMDPL_t = Planck15.age(SMDPL_z).value

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(SMDPL_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9
    _mah = np.maximum.accumulate(halos["mpeak_history_main_prog"], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))
        log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

    # From https://www.cosmosim.org/cms/simulations/smdpl/
    # The mass particle resolution is 9.63e7 Msun/h
    particle_mass_res = 9.63e7 / H_SMDPL
    # So we cut halos with M0 below 500 times the mass resolution.
    logmpeak_fit_min = np.log10(500 * particle_mass_res)
    logmpeak = log_mahs[:, -1]

    sel = logmpeak >= logmpeak_fit_min

    log_mahs = log_mahs[sel]
    log_smahs = log_smahs[sel]
    sfrh = sfrh[sel]
    halo_ids = halo_ids[sel]

    log_mah_fit_min = 10.0

    return halo_ids, log_mahs, log_smahs, sfrh, SMDPL_t, dt, log_mah_fit_min


def _write_collated_data_diffmah(outname, data, colnames):
    nrows, ncols = np.shape(data)
    assert len(colnames) == ncols, "data mismatched with header"
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


def _write_collated_data_diffstar(outname, data, colnames):
    nrows, ncols = np.shape(data)
    assert len(colnames) == ncols, "data mismatched with header"
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


def run_diffmah(inps):

    halo_ids_for_rank = inps[0]
    log_mahs_for_rank = inps[1]
    tarr = inps[2]
    lgm_min = inps[3]

    nstep = 150

    halo_ids_for_rank = halo_ids[indx]
    log_mahs_for_rank = log_mahs[indx]
    nhalos_for_rank = len(halo_ids_for_rank)

    output = []
    for i in range(nhalos_for_rank):
        halo_id = halo_ids_for_rank[i]
        lgmah = log_mahs_for_rank[i, :]

        p_init, loss_data = get_loss_data_diffmah(tarr, lgmah, lgm_min,)
        _res = jax_adam_wrapper(
            mse_loss_and_grads_diffmah, p_init, loss_data, nstep, n_warmup=1
        )
        p_best, loss_best, loss_arr, params_arr, fit_terminates = _res

        if fit_terminates == 1:
            outline = get_outline_diffmah(halo_id, loss_data, p_best, loss_best)
        else:
            outline = get_outline_bad_fit_diffmah(halo_id, lgmah[-1], TODAY)

        output.append(outline.strip().split())

    return output


def run_diffstar(inps):

    halo_ids_for_rank = inps[0]
    log_smahs_for_rank = inps[1]
    sfrhs_for_rank = inps[2]
    mah_params_for_rank = inps[3]
    logmp_for_rank = inps[4]
    tarr, dt = inps[5:7]
    kwargs = inps[7]

    nhalos_for_rank = len(halo_ids_for_rank)

    output = []
    for i in range(nhalos_for_rank):
        halo_id = halo_ids_for_rank[i]
        lgsmah = log_smahs_for_rank[i, :]
        sfrh = sfrhs_for_rank[i, :]
        mah_params = mah_params_for_rank[i]
        logmp_halo = logmp_for_rank[i]

        p_init, loss_data = get_loss_data_diffstar(
            tarr, dt, sfrh, lgsmah, logmp_halo, mah_params, **kwargs
        )
        _res = minimizer_wrapper(
            loss_func_diffstar, loss_func_deriv_diffstar, p_init, loss_data
        )
        p_best, loss_best, success = _res
        outline = get_outline_diffstar(halo_id, loss_data, p_best, loss_best, success)

    output.append(outline.strip().split())
    return output


header_diffstar = get_header_diffstar()
header_diffmah = get_header_diffmah()

colnames_diffstar = header_diffstar[1:].strip().split()
colnames_diffmah = header_diffmah[1:].strip().split()


if __name__ == "__main__":

    """
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    """

    # nranks = pool.comm.Get_size() - 1
    nranks = 100

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-outbase_diffstar", help="Basename of the output hdf5 file")
    parser.add_argument("-outbase_diffmah", help="Basename of the output hdf5 file")
    parser.add_argument("-indir", help="Input directory", default="BEBOP")
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
        "-ssfrh_floor", help="Clipping floor for sSFH", type=float, default=SSFRH_FLOOR,
    )

    args = parser.parse_args()

    indir = args.indir
    outbase_diffstar = args.outbase_diffstar
    outbase_diffmah = args.outbase_diffmah

    kwargs = {
        "fstar_tdelay": args.fstar_tdelay,
        "mass_fit_min": args.mass_fit_min,
        "ssfrh_floor": args.ssfrh_floor,
    }

    for subvol in range(576):

        _name_diffmah = outbase_diffmah + "_%d.h5" % subvol
        _name_diffstar = outbase_diffstar + "_%d.h5" % subvol
        _outpath_diffmah = os.path.join(args.outdir, _name_diffmah)
        _outpath_diffstar = os.path.join(args.outdir, _name_diffstar)
        _name_diffmah_exists = os.path.exists(_outpath_diffmah)
        _name_diffstar_exists = os.path.exists(_outpath_diffstar)

        print("Running subvol %d" % subvol)

        if _name_diffmah_exists and _name_diffstar_exists:
            continue

        start = time()
        _data = load_SMDPL_data(np.array([subvol]), data_drn=BEBOP_SMDPL)
        halo_ids, log_mahs, log_smahs, sfrhs, SMDPL_t, dt, log_mah_fit_min = _data

        nhalos_tot = len(halo_ids)
        _a = np.arange(0, nhalos_tot).astype("i8")
        indxs = np.array_split(_a, nranks)

        inputs_diffmah = []
        for indx in indxs:
            inputs_diffmah.append(
                [halo_ids[indx], log_mahs[indx], SMDPL_t, log_mah_fit_min]
            )

        _res_diffmah = run_diffmah(inputs_diffmah[0])
        # _res_diffmah = np.concatenate(pool.map(run_diffmah, inputs_diffmah), axis=0)

        _res_diffmah = _res_diffmah.astype(float)

        _write_collated_data_diffmah(_outpath_diffmah, _res_diffmah, colnames_diffmah)

        _res_diffmah = {
            key: val for (key, val) in zip(colnames_diffmah, _res_diffmah.T)
        }

        mah_fit_params = np.array(
            [
                _res_diffmah["mah_logtc"],
                _res_diffmah["mah_k"],
                _res_diffmah["early_index"],
                _res_diffmah["late_index"],
            ]
        ).T

        logmp = _res_diffmah["logmp_fit"]

        # """
        inputs_diffstar = []
        for indx in indxs:
            inputs_diffstar.append(
                [
                    halo_ids[indx],
                    log_smahs[indx],
                    sfrhs[indx],
                    mah_fit_params[indx],
                    logmp[indx],
                    SMDPL_t,
                    dt,
                    kwargs,
                ]
            )

        _res_diffstar = run_diffstar(inputs_diffstar[0])
        # _res_diffstar = np.concatenate(pool.map(run_diffstar, inputs_diffstar), axis=0)

        _res_diffstar = _res_diffstar.astype(float)

        _write_collated_data_diffstar(
            _outpath_diffstar, _res_diffstar, colnames_diffstar
        )

        end = time()

        msg = "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
        runtime = end - start
        print("Subvolume %d" % subvol)
        print(msg.format(nhalos_tot, nranks, runtime))
        # """
