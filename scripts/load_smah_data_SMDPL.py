"""
"""
import numpy as np
import os
import warnings
from umachine_pyio.load_mock import load_mock_from_binaries
from astropy.cosmology import Planck15
import h5py
from diffstar.utils import _get_dt_array


BEBOP_SMDPL = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_z0_binaries/"

H_BPL = 0.678


def load_fit_mah(filename, data_drn=BEBOP_SMDPL):
    """ Load the best fit diffmah parameter data.
    Parameters
    ----------
    filename : string
        Name of the h5 file where the diffmah best fit parameters are stored.
    data_drn : string
        Filepath where the Diffstar best-fit parameters are stored.

    Returns
    -------
    mah_fit_params:  ndarray of shape (n_gal, 4)
        Best fit parameters for each halo:
            (logtc, k, early_index, late_index)
    logmp:  ndarray of shape (n_gal, )
        Base-10 logarithm of the present day peak halo mass.
    """
    fitting_data = dict()

    fn = os.path.join(data_drn, filename)
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            if key == "halo_id":
                fitting_data[key] = hdf[key][...]
            else:
                fitting_data["fit_" + key] = hdf[key][...]

    mah_fit_params = np.array(
        [
            fitting_data["fit_mah_logtc"],
            fitting_data["fit_mah_k"],
            fitting_data["fit_early_index"],
            fitting_data["fit_late_index"],
        ]
    ).T
    logmp = fitting_data["fit_logmp_fit"]

    return mah_fit_params, logmp


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

    galprops = ["halo_id", "sfr_history_main_prog"]
    halos = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)

    SMDPL_a = np.load("/lcrc/project/halotools/UniverseMachine/SMDPL/scale_list.npy")
    SMDPL_z = 1.0 / SMDPL_a - 1.0
    SMDPL_t = Planck15.age(SMDPL_z).value

    halo_ids = halos["halo_id"]
    dt = _get_dt_array(SMDPL_t)
    sfrh = halos["sfr_history_main_prog"]
    sm_cumsum = np.cumsum(sfrh * dt, axis=1) * 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_smahs = np.where(sm_cumsum == 0, 0, np.log10(sm_cumsum))

    return halo_ids, log_smahs, sfrh, SMDPL_t, dt
